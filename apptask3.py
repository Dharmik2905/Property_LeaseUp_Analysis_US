import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
#for gemini ai
import google.generativeai as genai
import os
from dotenv import load_dotenv


st.set_page_config(
    page_title="Property Cluster Analysis",
    page_icon="🏢",
    layout="wide"
)


# Loading my  data
@st.cache_data
def load_data():
    # Load my CSV files
    merged_df = pd.read_csv("merged.csv")
    summary_df = pd.read_csv("summary.csv")
    
    # Define cluster names defined again, can be directly used later even we have column for it
    cluster_names = {
        0: "High-End New Developments",
        1: "Stable Established Properties",
        2: "New Development Pre-Lease",
        3: "Repositioning Properties",
        4: "Value-Oriented Stable Assets"
    }
    
    # Adding cluster names to dataframe
    merged_df['Cluster_Name'] = merged_df['Cluster'].map(cluster_names)
    
    return merged_df, summary_df


def generate_ai_insights(data):
    """Generate insights using Gemini API based on filtered dashboard data"""
    try:
        # First try Streamlit secrets which i have (for deployment)
        api_key = st.secrets["GEMINI_API_KEY"]
    except:
        try:
            load_dotenv()
            api_key = os.getenv("GEMINI_API_KEY")
        except:
            return "‼️ Gemini API key not found. Please set up your API key to enable AI insights."
    
    if not api_key:
        return "‼️ Gemini API key not found. Please set up your API key to enable AI insights."
    
    # Configuring the API
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    # Creating a prompt that asks for specific insights
    prompt = f"""
    You are a real estate analytics expert analyzing rental property data. 
    Based on the following filtered dataset, provide 3-5 key insights that would be valuable for investors or property managers. 
    Explain any real estate terms in plain language and fun language
    
    Data Summary:
    - Total Properties: {data['total_properties']}
    - Markets: {', '.join(data['markets'])}
    - Clusters: {', '.join(data['clusters'])}
    - Average Rent: ${data['avg_rent']:.2f}
    - Average Occupancy: {data['avg_occupancy']:.1f}%
    - Rent Range: ${data['rent_range'][0]:.2f} to ${data['rent_range'][1]:.2f}
    - Occupancy Range: {data['occupancy_range'][0]:.1f}% to {data['occupancy_range'][1]:.1f}%
    - Property Age Range: {data['property_age_range'][0]:.1f} to {data['property_age_range'][1]:.1f} years
    
    Cluster Distribution:
    {data['cluster_distribution']}
    
    Market Distribution:
    {data['market_distribution']}
    For your analysis:
    1. Identify any meaningful correlations or patterns in the data
    2. Compare performance across different clusters and markets
    3. Highlight any anomalies or outliers worth investigating
    4. Consider how property age relates to performance metrics

    Format your response as:
    - Clear bullet points with concise insights
    - Brief explanation of any technical terms
    - End with ONE actionable recommendation based on your analysis

    Your insights should be accessible to someone with basic real estate knowledge but not necessarily expertise in analytics.
    """
    
    try:
        # Generate the  insights based upon my filtered data for diff filters selected
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating insights: {str(e)}"



def main():
    # Loading data merged and summary df from Task 1
    merged_df, summary_df = load_data()
    
    # Title 
    st.title("🏢 Rental Property Cluster Analysis Dashboard")
    st.markdown("Interactive analysis of rental properties across different markets and clusters")
    
    # Sidebar filters on left side
    st.sidebar.title("Filters")
    
    # Market filter on left side ( 1st filter)
    markets = sorted(merged_df['MarketName'].unique())
    selected_markets = st.sidebar.multiselect(
        "Select Markets", 
        options=markets,
        default=markets[:2] if len(markets) > 1 else markets
    )
    
    # Cluster filter (second filter)
    clusters = sorted(merged_df['Cluster'].unique())
    selected_clusters = st.sidebar.multiselect(
        "Select Clusters", 
        options=clusters,
        default=clusters,
        format_func=lambda x: f"Cluster {x}: {merged_df[merged_df['Cluster'] == x]['Cluster_Name'].iloc[0]}"
    )
    
    # Submarket filter (optional more filter)
    submarkets_all = sorted(merged_df['Submarket'].unique())
    selected_submarkets = st.sidebar.multiselect(
        "Select Submarkets (Optional)", 
        options=submarkets_all,
        default=[]
    )
    
    # Applying filters
    if selected_submarkets:
        filtered_df = merged_df[
            (merged_df['MarketName'].isin(selected_markets)) & 
            (merged_df['Cluster'].isin(selected_clusters)) &
            (merged_df['Submarket'].isin(selected_submarkets))
        ]
    else:
        filtered_df = merged_df[
            (merged_df['MarketName'].isin(selected_markets)) & 
            (merged_df['Cluster'].isin(selected_clusters))
        ]
    
    # Making sure we got data if we selected a specific filter
    if filtered_df.empty:
        st.warning("No data available with the selected filters.")
        return
    
    # Displaying the basic statistics - Key Metrics/KPIs 
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Properties", filtered_df.shape[0])
    
    with col2:
        st.metric("Average Rent", f"${filtered_df['Avg_Effective_Rent'].mean():.2f}")
    
    with col3:
        st.metric("Average Occupancy", f"{filtered_df['Avg_Occupancy'].mean()*100:.1f}%")
    
    with col4:
        st.metric("Average Property Age", f"{filtered_df['Property_Age'].mean():.1f} years")
    
    # Map Section
    st.header("Geographic Distribution")
    
    # Create map with property locations - map makes it attractive
    fig = px.scatter_mapbox(
        filtered_df,
        lat="Latitude",
        lon="Longitude",
        color="Cluster_Name",
        size="Estimated_Property_Size",
        hover_name="Name",
        hover_data={
            "MarketName": True,
            "Submarket": True,
            "Avg_Effective_Rent": ":.2f",
            "Avg_Occupancy": ":.2%",
            "Property_Age": ":.1f",
            "YearBuilt": True,
            "Estimated_Property_Size": True
        },
        zoom=5,
        height=600,
        size_max=15,
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    
    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Rent vs. Occupancy", "Property Metrics", "Cluster Analysis", "Market Comparison"])
    
    with tab1:
        st.subheader("Average Effective Rent vs. Occupancy")
        
        # Create scatter plot
        fig = px.scatter(
            filtered_df,
            x="Avg_Effective_Rent",
            y="Avg_Occupancy",
            color="Cluster_Name",
            size="Estimated_Property_Size",
            hover_name="Name",
            hover_data={
                "MarketName": True,
                "Submarket": True,
                "Avg_Effective_Rent": ":.2f",
                "Avg_Occupancy": ":.2%",
                "Property_Age": ":.1f",
                "Avg_Concession": ":.1f"
            },
            size_max=25,
            opacity=0.7,
            height=500,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        
        fig.update_layout(
            xaxis_title="Average Effective Rent ($)",
            yaxis_title="Average Occupancy Rate",
            legend_title="Cluster"
        )
        
        fig.update_yaxes(tickformat=".0%")
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Property Metrics Distribution")
        
        # Select which metric to display for drill down
        metric_options = {
            "Avg_Effective_Rent_Per_SqFt": "Effective Rent Per SqFt ($)",
            "Property_Age": "Property Age (Years)",
            "Avg_Concession": "Average Concession (Days)",
            "Lease_Up_Months": "Lease-Up Months",
            "Estimated_Property_Size": "Property Size (SqFt)"
        }
        
        selected_metric = st.selectbox(
            "Select Metric", 
            options=list(metric_options.keys()),
            format_func=lambda x: metric_options[x]
        )
        
        # Create histogram with boxplot to undertstand the spread
        fig = px.histogram(
            filtered_df,
            x=selected_metric,
            color="Cluster_Name",
            marginal="box",
            barmode="overlay",
            opacity=0.7,
            height=400,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        
        fig.update_layout(
            xaxis_title=metric_options[selected_metric]
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Cluster Characteristics")
        
        # Calculate cluster averages for the filtered data helps to check the cluster properties
        cluster_summary = filtered_df.groupby('Cluster_Name').agg({
            'Avg_Effective_Rent': 'mean',
            'Avg_Occupancy': 'mean',
            'Property_Age': 'mean',
            'Lease_Up_Months': lambda x: x.replace(0, np.nan).mean(),
            'Avg_Concession': 'mean',
            'Avg_Effective_Rent_Per_SqFt': 'mean',
            'Cluster': 'count'
        }).rename(columns={'Cluster': 'Count'}).reset_index()
        
        # Create bar chart with dropdown for different metrics, simple visualisation and comparision
        fig = go.Figure()
        
        metrics = ['Avg_Effective_Rent', 'Avg_Occupancy', 'Property_Age', 'Avg_Concession']
        titles = ['Avg. Effective Rent ($)', 'Avg. Occupancy Rate', 'Property Age (Years)', 'Avg. Concession (Days)']
        
        # Add dropdown menu for different metrics, for more details
        fig.update_layout(
            updatemenus=[
                dict(
                    active=0,
                    buttons=list([
                        dict(label=titles[i],
                        method="update",
                        args=[{"y": [[cluster_summary[metric][j] for j in range(len(cluster_summary))]]},
                            {"yaxis.title.text": titles[i]}]) for i, metric in enumerate(metrics)
                    ]),
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=1.15,
                    yanchor="top"
                ),
            ]
        )
        
        # Initial visualization
        fig.add_trace(go.Bar(
            x=cluster_summary['Cluster_Name'],
            y=cluster_summary['Avg_Effective_Rent'],
            text=cluster_summary['Count'].apply(lambda x: f"{x} properties"),
            textposition='auto',
            marker_color=px.colors.qualitative.Bold[:len(cluster_summary)]
        ))
        
        fig.update_layout(
            xaxis_title="Cluster",
            yaxis_title=titles[0],
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display summary data
        st.subheader("Detailed Cluster Metrics")
        st.dataframe(
            cluster_summary.style.format({
                'Avg_Effective_Rent': '${:.2f}',
                'Avg_Occupancy': '{:.1%}',
                'Property_Age': '{:.1f} yrs',
                'Avg_Concession': '{:.1f}$',
                'Avg_Effective_Rent_Per_SqFt': '${:.2f}',
                'Count': '{:.0f}'
            }),
            use_container_width=True
        )
    
    with tab4:
        st.subheader("Market Comparison")
        
        # Market comparison
        market_summary = filtered_df.groupby('MarketName').agg({
            'Avg_Effective_Rent': 'mean',
            'Avg_Occupancy': 'mean',
            'Property_Age': 'mean',
            'Lease_Up_Months': lambda x: x.replace(0, np.nan).mean(),
            'Avg_Concession': 'mean',
            'Cluster': 'count'
        }).rename(columns={'Cluster': 'Property Count'}).reset_index()
        # Calculating the correct Property Percentage based on selected filters
        total_properties = market_summary["Property Count"].sum()
        market_summary["Property Percentage"] = (market_summary["Property Count"] / total_properties) * 100
        # Displaying the market summary
        st.dataframe(
            market_summary.style.format({
                'Avg_Effective_Rent': '${:.2f}',
                'Avg_Occupancy': '{:.1%}',
                'Property_Age': '{:.1f} yrs',
                'Lease_Up_Months': '{:.1f}',
                'Avg_Concession': '{:.1f}$',
                'Property Count': '{:.2f}',
                'Property Percentage':'{:.2f}%'
            }),
            use_container_width=True
        )
        
        # Market cluster distribution, helps to know 5 clusters from Task 1 in specific Market
        st.subheader("Cluster Distribution by Market")
        
        # Calculate cluster distribution by market
        cluster_dist = pd.crosstab(
            filtered_df['MarketName'],
            filtered_df['Cluster_Name'],
            normalize='index'
        ) * 100
        
        # Create stacked bar chart
        fig = px.bar(
            cluster_dist.reset_index().melt(id_vars=['MarketName'], var_name='Cluster', value_name='Percentage'),
            x='MarketName',
            y='Percentage',
            color='Cluster',
            barmode='stack',
            height=400,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        
        fig.update_layout(
            xaxis_title="Market",
            yaxis_title="Percentage of Properties (%)",
            legend_title="Cluster"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    # Add AI Insights Section
    st.header("🤔Insights Explanation: Want a brief summary. Click the button below!")
    st.markdown("Get AI-generated 🤖 insights based on the available filtered data displayed on dashboard")

    # Creating a button to generate insights based upon current one
    if st.button("Generate AI Insights"):
        with st.spinner("Analyzing your data..."):
        # Prepare data summary for the AI
            insight_data = {
                "total_properties": filtered_df.shape[0],
                "markets": filtered_df['MarketName'].unique().tolist(),
                "clusters": filtered_df['Cluster_Name'].unique().tolist(),
                "avg_rent": filtered_df['Avg_Effective_Rent'].mean(),
                "avg_occupancy": filtered_df['Avg_Occupancy'].mean() * 100,
                "rent_range": [filtered_df['Avg_Effective_Rent'].min(), filtered_df['Avg_Effective_Rent'].max()],
                "occupancy_range": [filtered_df['Avg_Occupancy'].min() * 100, filtered_df['Avg_Occupancy'].max() * 100],
                "property_age_range": [filtered_df['Property_Age'].min(), filtered_df['Property_Age'].max()],
                "cluster_distribution": filtered_df['Cluster_Name'].value_counts().to_dict(),
                "market_distribution": filtered_df['MarketName'].value_counts().to_dict()
            }
            
            # Generate AI insights (implement the function below)
            insights = generate_ai_insights(insight_data)
            
            # Display insights
            st.markdown("### Key Insights")
            st.markdown(insights)

# Run the app
if __name__ == "__main__":
    main()
#Looks great.
#Used Streamlit for quick Dashboarding, other can be Tableau from our updated CSVs
#Avoided inbuilt jupyter dashboard, as it sounds trivial
#took bit help of ChatGPT and youtube for making this dashboard -> Though I Got the overall format , structure
#and layout of my dashbaord, and upgraded my code eventually as and when I wanted specific functionalities
