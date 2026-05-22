import os
from crewai import LLM, Agent, Task, Crew, Process

os.environ["DEEPSEEK_API_KEY"] = "your-deepseek-api-key-here"

def run_crew():
    deepseek_llm = LLM(
        model="deepseek/deepseek-chat",
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        temperature=0.7
    )

    researcher = Agent(
        role="Senior Tech Researcher",
        goal="Uncover groundbreaking advancements in AI and automation.",
        backstory="You are a meticulous researcher at a top think-tank. You excel at parsing complex technical developments and distilling them into core facts.",
        llm=deepseek_llm,
        verbose=True,
        allow_delegation=False
    )

    writer = Agent(
        role="Tech Journalist",
        goal="Create engaging, easy-to-digest articles about complex technology.",
        backstory="You are a writer for a major tech publication. You know how to take raw research data and turn it into a compelling story that captivates audiences.",
        llm=deepseek_llm,
        verbose=True,
        allow_delegation=False
    )

    research_task = Task(
        description="Analyze the current state of open-source LLMs in early 2026. Focus on efficiency breakthroughs.",
        expected_output="A bulleted list of the top 3 breakthroughs with brief technical explanations.",
        agent=researcher
    )

    write_task = Task(
        description="Using the research provided, write a short, punchy 3-paragraph blog post for a tech-savvy audience.",
        expected_output="A complete, well-formatted 3-paragraph markdown blog post.",
        agent=writer
    )

    tech_crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        process=Process.sequential,
        verbose=True
    )

    result = tech_crew.kickoff()
    print(result)

if __name__ == "__main__":
    run_crew()