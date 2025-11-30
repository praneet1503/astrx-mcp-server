---
title: Astrx Mcp Server - Hackathon Space Submission
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
short_description: Sample space with tags for MCP 1st Birthday party
tags:
  - building-mcp-track-consumer
---
## 🤖 Automation Pipeline

This repository includes a GitHub Action workflow to automate data scraping and deployment.

### Setup

1. **Modal Secrets**:
   - Go to your GitHub Repository Settings > Secrets and variables > Actions.
   - Add the following secrets:
     - `MODAL_TOKEN_ID`: Your Modal Token ID.
     - `MODAL_TOKEN_SECRET`: Your Modal Token Secret.

2. **Hugging Face Sync (Optional)**:
   - If you want to sync the data to a Hugging Face Space automatically:
     - Add `HF_TOKEN`: Your Hugging Face Write Token.
     - `HF_SPACE_REPO`: The repository ID of your Space (e.g., `username/space-name`).

3. **Hugging Face Space Configuration**:
   - Go to your Space's **Settings** tab.
   - Scroll down to **Variables and secrets**.
   - Click **New secret**.
   - Name: `CLAUDE_API_KEY`
   - Value: Your Anthropic API Key (starts with `sk-ant-...`).

### Workflow

The workflow runs automatically every Sunday at midnight or can be triggered manually.
1. Runs the Modal scraper.
2. Validates the generated `animals.json`.
3. Archives the data to `data/versions/`.
4. Commits and pushes the changes to the repository.
5. (Optional) Uploads the data to the configured HF Space.
