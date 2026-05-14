# Agentic Forecasting Platform

A modular system combining:
- Time-series forecasting models
- Data pipelines
- AI agents for orchestration
- API serving layer

## Architecture

data_pipeline/ → ingestion + preprocessing  
models/ → forecasting models  
agents/ → decision-making agents  
orchestration/ → workflow logic  
api/ → FastAPI serving layer  
utils/ → shared utilities  
notebooks/ → experiments

## Goal

Build a production-grade forecasting system with agent-based decision layers.

## How to run

Always run modules from project root:

python3 -m data_pipeline.ingestion