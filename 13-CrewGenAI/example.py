from crewai import Agent, Task, Crew
from crewai_tools import YoutubeChannelSearchTool

# Initialize the tool
youtube_channel_tool = YoutubeChannelSearchTool()

# Define an agent that uses the tool
channel_researcher = Agent(
    role="Channel Researcher",
    goal="Extract and analyze information from YouTube channels",
    backstory="""You are an expert channel researcher who specializes in extracting 
    and analyzing information from YouTube channels. You have a keen eye for detail 
    and can quickly identify key points and insights from video content across an entire channel.""",
    tools=[youtube_channel_tool],
    verbose=True,
)

# Create a task for the agent
research_task = Task(
    description="""
    Search for information about data science projects and tutorials 
    in the YouTube channel {youtube_channel_handle}. 

    Focus on:
    1. Key data science techniques covered
    2. Popular tutorial series
    3. Most viewed or recommended videos

    Provide a comprehensive summary of these points.
    """,
    expected_output="A detailed summary of data science content available on the channel.",
    agent=channel_researcher,
)

# Run the task
crew = Crew(agents=[channel_researcher], tasks=[research_task])
result = crew.kickoff(inputs={"youtube_channel_handle": "@codebasics"})