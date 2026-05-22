import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai.flow.flow import Flow, listen, start, or_

load_dotenv()

from crewai import LLM

groq_llm = LLM(
    model="groq/llama3-8b-8192",
    api_key=os.getenv("GROQ_API_KEY"),
)
class SimpleMarketingFlow(Flow):

    @start()
    def generate_product_idea(self):
        print("[Step 1] Generating a product idea...")
        self.state["product"] = "Solar Powered Coffee Mug"
        return self.state["product"]

    @start()
    def secondary_trigger(self):
        print("[Step 1b] Parallel trigger running...")
        return "Bonus feature: LED Screen"

    @listen(generate_product_idea)
    def run_crew_analysis(self, product_name):
        print(f"[Step 2] Running a Crew to analyze: '{product_name}'")
        
        analyst = Agent(
            role="Product Marketer",
            goal="Create a 1-sentence catchy tagline",
            backstory="An expert in viral marketing trends.",
            llm=groq_llm
        )
        
        tagline_task = Task(
            description=f"Create a funny tagline for a {product_name}.",
            expected_output="A single catchy and humorous sentence.",
            agent=analyst
        )
        
        my_crew = Crew(agents=[analyst], tasks=[tagline_task])
        result = my_crew.kickoff()
        
        self.state["tagline"] = result.raw
        return self.state["tagline"]

    @listen(or_(generate_product_idea, secondary_trigger))
    def conditional_logger(self):
        print("[Notice] Either the main product or the bonus feature was initialized!")

    @listen(run_crew_analysis)
    def final_wrap_up(self, final_tagline):
        print("\n=== FINAL WORKFLOW RESULT ===")
        print(f"Product: {self.state.get('product')}")
        print(f"Tagline: {final_tagline}")


if __name__ == "__main__":
    flow = SimpleMarketingFlow()
    flow.kickoff()