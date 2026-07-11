# Project Structure & Directory Guide

Welcome to the **Hybrid Token-Efficient Routing Agent**! This guide is written specifically for newcomers to Python, Artificial Intelligence (AI), and Machine Learning (ML). It breaks down the files in this repository, explains how they interact, and defines key terms you'll encounter.

---

## 1. What This Project Does: The High-Level Idea

Large AI models (like ChatGPT or Claude) are smart but **expensive** because you pay a cloud provider for every word they generate. Small AI models (which you can run directly on your own computer) are **free**, but they are not as smart and make more mistakes.

This project builds a **Hybrid Routing System**:
1. When a user asks a question, we first ask a small, free **local model** running on our system.
2. While generating the answer, we extract the local model's **confidence scores** (how sure it is about each word it wrote).
3. We check those confidence scores against a threshold:
   * **If the confidence is high:** We keep the local model's answer (Cost = $0).
   * **If the confidence is low:** We throw away the local answer and escalate the question to a larger, paid **cloud model** (Fireworks AI) to get a correct response.

This workflow saves money while keeping the final accuracy high!

---

## 2. Visual Flow (How the Files Talk to Each Other)

Here is a simplified flowchart of what happens when you run a question through the system:

```text
               [ User asks a question ]
                          |
                          v
                    agent_loop.py (The Coordinator)
                          |
             +------------+------------+
             |                         |
             v                         v
     local_model.py             remote_client.py
     (Runs Qwen/Gemma           (Calls Fireworks AI
      locally for free)          cloud model - costs money)
             |                         |
             +------------+------------+
                          |
                          v
                      router.py (The Decision Maker)
                 (Compares confidence to thresholds)
                          |
                          v
                   [ Final Answer ]
                          |
                          v
         Runs directory: appends logs to agent_log.jsonl
```

---

## 3. Detailed File-by-File Breakdown

### Core Code (Python Scripts)
*   **[agent_loop.py](file:///Users/adityatiwari/Dev/Hybrid-Token-Efficient-Routing-Agent-/agent_loop.py) (The Coordinator)**
    *   **What it does:** This is the main entry point to run the app. It takes your prompt, passes it to the local model runner, asks the router whether to keep the answer or ask the cloud model, and then saves a log of the final decision.
*   **[local_model.py](file:///Users/adityatiwari/Dev/Hybrid-Token-Efficient-Routing-Agent-/local_model.py) (The Local AI Runner)**
    *   **What it does:** Loads the small AI model (e.g., Qwen or Gemma) onto your computer's memory and tells it to generate a response. Crucially, it extracts the math statistics (confidence scores) for each word generated.
*   **[router.py](file:///Users/adityatiwari/Dev/Hybrid-Token-Efficient-Routing-Agent-/router.py) (The Filter/Decision Maker)**
    *   **What it does:** Contains the rules for routing. It checks if the average confidence (`mean_logprob`) and the single lowest confidence token (`min_logprob`) are above your settings. If either score is too low, it triggers an escalation.
*   **[remote_client.py](file:///Users/adityatiwari/Dev/Hybrid-Token-Efficient-Routing-Agent-/remote_client.py) (The Cloud AI Connector)**
    *   **What it does:** Connects to the cloud AI service (Fireworks AI) over the internet to get a high-quality answer. This script is only executed if the router decides to escalate.
*   **[config.py](file:///Users/adityatiwari/Dev/Hybrid-Token-Efficient-Routing-Agent-/config.py) (The Control Panel)**
    *   **What it does:** Reads settings (like model names, API keys, and threshold values) from your environment and local files, acting as a single place to configure the app.
*   **[calibrate.py](file:///Users/adityatiwari/Dev/Hybrid-Token-Efficient-Routing-Agent-/calibrate.py) (The Tuner/Optimizer)**
    *   **What it does:** Helps you find the best threshold settings. It runs the local model over all the test questions, grades how many it got right, and sweeps through different threshold settings to find the one that escalates the least while maintaining high accuracy.
*   **[generate_tasks.py](file:///Users/adityatiwari/Dev/Hybrid-Token-Efficient-Routing-Agent-/generate_tasks.py) (The Question Creator)**
    *   **What it does:** Auto-generates arithmetic questions with exact numerical answers so we have a large test suite without having to write hundreds of questions manually.

### Data & Output Files
*   **[tasks_factual.jsonl](file:///Users/adityatiwari/Dev/Hybrid-Token-Efficient-Routing-Agent-/tasks_factual.jsonl) (Trivia Questions)**
    *   **What it is:** A text dataset of hand-picked factual and reasoning questions (like "What is the capital of France?").
*   **[tasks.jsonl](file:///Users/adityatiwari/Dev/Hybrid-Token-Efficient-Routing-Agent-/tasks.jsonl) (Master Test Suite)**
    *   **What it is:** The full set of test questions, combining `tasks_factual.jsonl` with the auto-generated math questions.
*   **`runs/` (Output Folder)**
    *   `agent_log.jsonl`: A live diary recording every single question you ask, the route taken, confidence scores, cost, and time taken.
    *   `calibration_results.jsonl`: The detailed raw answers and grades for every test question during a calibration run.
    *   `calibration_sweep.csv`: A spreadsheet showing what the cost and accuracy would be for every potential threshold setting you could choose.

### Project Settings & Setup
*   **[.gitignore](file:///Users/adityatiwari/Dev/Hybrid-Token-Efficient-Routing-Agent-/.gitignore) (Version Control Ignore List)**
    *   **What it does:** Tells Git (the code version tracker) which files/folders to ignore. This keeps your repository clean and prevents you from uploading huge virtual environments (like `.venv`) or private passwords.
*   **[requirements.txt](file:///Users/adityatiwari/Dev/Hybrid-Token-Efficient-Routing-Agent-/requirements.txt) (Package List)**
    *   **What it does:** A shopping list of external Python libraries that this project needs to run (e.g., code to make web requests or parse data).
*   **[Dockerfile](file:///Users/adityatiwari/Dev/Hybrid-Token-Efficient-Routing-Agent-/Dockerfile) (Deployment Blueprint)**
    *   **What it does:** A recipe file to package this entire codebase into an isolated "container" so it can run on any server or machine in the cloud exactly as it does on your computer.

---

## 4. Key Terminology Guide

If you are new to AI/ML, here are some terms used frequently in this codebase:

*   **Token:** AI models don't read text word-by-word. Instead, they chop words into smaller chunks called "tokens" (e.g., "eating" might become "eat" and "ing").
*   **Logprob (Log-probability):** A mathematical way to measure how confident the model is when picking a token. 
    *   A logprob of `0.0` means **100% confident**.
    *   Logprobs are always **negative numbers** (e.g., `-0.2` is high confidence, while `-4.5` is extremely low confidence/uncertainty).
*   **Mean Logprob vs Min Logprob:**
    *   *Mean logprob* is the average confidence across all words in the answer.
    *   *Min logprob* is the confidence score of the single most uncertain word. This is super helpful because it catches cases where the AI is fluent and confident in general, but completely guesses a specific number or date (a "hallucination").
*   **JSONL / JSON Lines:** A text format where each line is a standalone structured database record. It's popular in AI/ML because it's very fast to read line-by-line.
*   **Mock Mode:** A built-in feature in this repository. By running commands with `MOCK_LOCAL_MODEL=1`, you tell the code to simulate AI generation using fake, pre-programmed answers. This lets you test the code instantly on a normal laptop without downloading heavy AI models or paying for API keys.
