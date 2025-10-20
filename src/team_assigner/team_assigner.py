from sklearn.cluster import KMeans
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.player_team_history = {}  # Track assignment history
        
        # Configuration
        self.confidence_threshold = 0.6  # Minimum confidence to assign
        self.hysteresis_frames = 5  # Consecutive frames to confirm switch
        self.min_color_separation = 50.0  # Minimum RGB distance between teams
    
    def get_clustering_model(self,image):
        # Reshape the image to 2D array
        image_2d = image.reshape(-1,3)

        # Preform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self,frame,bbox):
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

        top_half_image = image[0:int(image.shape[0]/2),:]

        # Get Clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels forr each pixel
        labels = kmeans.labels_

        # Reshape the labels to the image shape
        clustered_image = labels.reshape(top_half_image.shape[0],top_half_image.shape[1])

        # Get the player cluster
        corner_clusters = [clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters),key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def get_color_distance(self, color1, color2):
        """Calculate Euclidean distance between two RGB colors"""
        return np.linalg.norm(np.array(color1) - np.array(color2))
    
    def get_team_confidence(self, player_color):
        """
        Calculate confidence [0-1] for team assignment.
        Higher confidence = player color is much closer to one team than the other.
        """
        if not self.team_colors or len(self.team_colors) < 2:
            return 0.0
        
        dist1 = self.get_color_distance(player_color, self.team_colors[1])
        dist2 = self.get_color_distance(player_color, self.team_colors[2])
        
        total_dist = dist1 + dist2
        if total_dist == 0:
            return 0.5  # Ambiguous
        
        # Confidence based on distance ratio
        min_dist = min(dist1, dist2)
        max_dist = max(dist1, dist2)
        
        # Normalize: 0 when equal, 1 when one is 0
        confidence = (max_dist - min_dist) / total_dist
        
        return confidence

    def validate_team_colors(self):
        """
        Validate that team colors are sufficiently different.
        
        Returns:
            bool: True if colors are well-separated, False if too similar
        """
        if not self.team_colors or len(self.team_colors) < 2:
            logger.warning("Team colors not initialized")
            return False
        
        color_diff = self.get_color_distance(
            self.team_colors[1],
            self.team_colors[2]
        )
        
        if color_diff < self.min_color_separation:
            logger.warning(
                f"Team colors too similar (distance={color_diff:.1f}, "
                f"threshold={self.min_color_separation}). "
                f"Team 1: {self.team_colors[1]}, Team 2: {self.team_colors[2]}"
            )
            return False
        
        logger.info(f"Team color separation: {color_diff:.1f} (good)")
        return True

    def assign_team_color(self,frame, player_detections):
        """
        Initialize team colors with robust K-Means clustering.
        
        Improvements:
        - More initializations (n_init=20) for stability
        - Fixed random seed for reproducibility
        - Validation that team colors are sufficiently different
        """
        
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color =  self.get_player_color(frame,bbox)
            player_colors.append(player_color)
        
        logger.info(f"Initializing team colors with {len(player_colors)} players")
        
        # Robust K-Means with multiple initializations
        kmeans = KMeans(
            n_clusters=2,
            init="k-means++",  # Smart initialization
            n_init=20,  # Multiple runs to find best
            max_iter=300,
            random_state=42  # Reproducibility
        )
        kmeans.fit(player_colors)

        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]
        
        # Log team colors for debugging
        logger.info(f"Team 1 color (RGB): {self.team_colors[1]}")
        logger.info(f"Team 2 color (RGB): {self.team_colors[2]}")
        
        # Validate color separation
        if not self.validate_team_colors():
            logger.warning("Team colors may be too similar; assignments might be unreliable")


    def get_player_team(self,frame,player_bbox,player_id):
        """
        Assign player to team with hysteresis to prevent flickering.
        
        Logic:
        - If first time seeing player: assign based on color
        - If seen before and confidence is low: keep previous assignment
        - If predicting different team: require N consecutive frames to confirm
        """
        
        # Check if kmeans is initialized
        if not hasattr(self, 'kmeans') or self.kmeans is None:
            logger.warning("KMeans not initialized, cannot assign team")
            return -1
        
        # Get player's current jersey color
        player_color = self.get_player_color(frame, player_bbox)
        
        # Calculate confidence in assignment
        confidence = self.get_team_confidence(player_color)
        
        # Predict team based on color
        predicted_team = self.kmeans.predict(player_color.reshape(1, -1))[0] + 1
        
        # Check if player has history
        if player_id in self.player_team_history:
            history = self.player_team_history[player_id]
            current_team = history['team']
            
            # Low confidence → maintain previous assignment
            if confidence < self.confidence_threshold:
                logger.debug(f"Player {player_id}: low confidence ({confidence:.2f}), keeping team {current_team}")
                history['low_confidence_count'] = history.get('low_confidence_count', 0) + 1
                return current_team
            else:
                history['low_confidence_count'] = 0
            
            # Predicted team differs from current → require hysteresis
            if predicted_team != current_team:
                history['switch_counter'] = history.get('switch_counter', 0) + 1
                
                # Not enough consecutive frames to confirm switch
                if history['switch_counter'] < self.hysteresis_frames:
                    logger.debug(f"Player {player_id}: switch pending ({history['switch_counter']}/{self.hysteresis_frames}), keeping team {current_team}")
                    return current_team
                else:
                    # Switch confirmed
                    logger.info(f"Player {player_id}: team switch {current_team} → {predicted_team} (confidence={confidence:.2f})")
                    history['team'] = predicted_team
                    history['switch_counter'] = 0
                    history['confidence'] = confidence
                    self.player_team_dict[player_id] = predicted_team
                    return predicted_team
            else:
                # Prediction matches current team → reset switch counter
                history['switch_counter'] = 0
                history['confidence'] = confidence
                return current_team
        
        else:
            # First time seeing this player
            logger.info(f"Player {player_id}: initial assignment to team {predicted_team} (confidence={confidence:.2f})")
            self.player_team_history[player_id] = {
                'team': predicted_team,
                'confidence': confidence,
                'switch_counter': 0,
                'low_confidence_count': 0
            }
            self.player_team_dict[player_id] = predicted_team
            return predicted_team
