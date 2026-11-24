"""Minimal two-agent demo: one GPU model, three calls, separate agent histories.

Teammates can import LocalLLM from model_pipeline in their own experiments.
"""
import os

from model_pipeline import LocalLLM


class Agent:
    def __init__(self, instructions, llm):
        self.llm = llm                     # Both agents share ONE loaded model.
        self.history = [{"role": "system", "content": instructions}]

    def ask(self, text):
        self.history.append({"role": "user", "content": text})
        answer = self.llm.generate(self.history)
        self.history.append({"role": "assistant", "content": answer})
        return answer


def main():
    # Change the model choice with MODEL=qwen4 / qwen8 / deepseek8 / moonlight16.
    llm = LocalLLM(model_name=os.environ.get("MODEL", "qwen8"))
    solver = Agent("Suggest practical solutions in a few sentences.", llm)
    critic = Agent("Identify weaknesses and suggest improvements.", llm)

    proposal = solver.ask("Suggest a way to reduce traffic near a university.")
    print("\nSOLVER:\n", proposal, flush=True)

    feedback = critic.ask(f"Review this proposal and point out problems:\n{proposal}")
    print("\nCRITIC:\n", feedback, flush=True)

    revised = solver.ask(f"Improve your previous proposal using this feedback:\n{feedback}")
    print("\nREVISED SOLUTION:\n", revised, flush=True)


if __name__ == "__main__":
    main()
