import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from dotenv import load_dotenv
from kneed import KneeLocator
import json

load_dotenv()
plt.style.use('seaborn')
pd.set_option('display.float_format', '{:,.2f}'.format)

class MarketSegmenter:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.scaler = StandardScaler()
        
    def preprocess_data(self):
        """Prepare data for clustering"""
        df = self.data.copy()
        
        df['revenue_per_user'] = df['revenue'] / df['users']
        df['growth_rate'] = df.groupby('region')['users'].pct_change() * 100
        
        region_df = df.groupby('region').agg({
            'growth_rate': 'mean',
            'revenue_per_user': 'mean',
            'users': 'sum'
        }).reset_index()
        
        features = ['growth_rate', 'revenue_per_user']
        region_df[features] = self.scaler.fit_transform(region_df[features])
        
        return region_df, features
    
    def determine_optimal_clusters(self, data):
        """Find optimal number of clusters using elbow method"""
        distortions = []
        K = range(1, 10)
        
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data)
            distortions.append(kmeans.inertia_)
            
        kl = KneeLocator(K, distortions, curve='convex', direction='decreasing')
        return kl.elbow
    
    def perform_clustering(self, data, features, n_clusters=None):
        """Perform K-means clustering"""
        if n_clusters is None:
            n_clusters = self.determine_optimal_clusters(data[features])
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        data['cluster'] = kmeans.fit_predict(data[features])
        
        # Get cluster centers in original scale
        centers = self.scaler.inverse_transform(kmeans.cluster_centers_)
        cluster_centers = pd.DataFrame(centers, columns=features)
        cluster_centers['cluster'] = range(n_clusters)
        
        return data, cluster_centers
    
    def visualize_clusters(self, data, features):
        """Create interactive cluster visualization"""
        fig = px.scatter(
            data,
            x=features[0],
            y=features[1],
            color='cluster',
            size='users',
            hover_name='region',
            title='Market Segmentation by Growth Rate and Revenue per User',
            labels={
                'growth_rate': 'Average Growth Rate (%)',
                'revenue_per_user': 'Revenue per User (USD)'
            }
        )
        
        return fig
    
    def generate_segment_profiles(self, data, cluster_centers):
        """Generate business insights for each segment"""
        segments = []
        
        for cluster in sorted(data['cluster'].unique()):
            cluster_data = data[data['cluster'] == cluster]
            center = cluster_centers[cluster_centers['cluster'] == cluster].iloc[0]
            
            segment = {
                'cluster': cluster,
                'growth_rate': center['growth_rate'],
                'revenue_per_user': center['revenue_per_user'],
                'regions': cluster_data['region'].tolist(),
                'size': cluster_data['users'].sum(),
                'characteristics': self._describe_segment(center)
            }
            
            segments.append(segment)
        
        return segments
    
    def _describe_segment(self, center):
        """Generate human-readable segment description"""
        if center['growth_rate'] > 1 and center['revenue_per_user'] > 100:
            return "High-growth, high-value markets (e.g., North America, Western Europe)"
        elif center['growth_rate'] > 1 and center['revenue_per_user'] <= 100:
            return "High-growth, emerging markets (e.g., Southeast Asia, Latin America)"
        elif center['growth_rate'] <= 1 and center['revenue_per_user'] > 100:
            return "Mature, high-value markets (e.g., Japan, Australia)"
        else:
            return "Developing markets with growth potential (e.g., Africa, Eastern Europe)"
    
    def run_analysis(self):
        """Run complete market segmentation pipeline"""
        data, features = self.preprocess_data()
        clustered_data, centers = self.perform_clustering(data, features)
        fig = self.visualize_clusters(clustered_data, features)
        segments = self.generate_segment_profiles(clustered_data, centers)
        
        # Save outputs
        clustered_data.to_csv('output/market_segments.csv', index=False)
        fig.write_html('output/segmentation_visualization.html')
        with open('output/segment_profiles.json', 'w') as f:
            json.dump(segments, f, indent=2)
        
        print(f"Identified {len(segments)} market segments")
        return segments

if __name__ == "__main__":
    segmenter = MarketSegmenter('data/digital_health_hmo_regions.csv')
    segments = segmenter.run_analysis()
