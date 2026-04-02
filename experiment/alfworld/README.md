# ALFWorld-LangGraph Integration

Zero-invasion experimental module for testing LangGraph Kernel System with ALFWorld dataset.

## Overview

This module provides a plug-and-play integration between ALFWorld (text-based household task environment) and LangGraph's Kernel System. It follows the **zero-invasion principle**: no modifications to `/home/syq/Documents/blackboard/langgraph/` files.

## Architecture

```
experiment/alfworld/
├── core/                      # Core integration components
│   ├── adapter.py            # ALFWorldKernelAdapter (main integration point)
│   └── state_bridge.py       # State conversion utilities
├── workers/                   # ALFWorld-specific workers (Phase 3)
│   ├── alfworld_architect.py # Task analysis and workflow generation
│   └── action_worker.py      # Action selection worker
├── environments/              # Environment wrappers
│   └── env_wrapper.py        # ALFWorld environment wrapper
├── evaluators/                # Evaluation tools (Phase 4)
├── utils/                     # Utility modules (Phase 4)
├── configs/                   # Configuration files (Phase 4)
├── examples/                  # Example scripts (Phase 4)
└── tests/                     # Unit tests
```

## Implementation Status

### ✅ Phase 1: Basic Infrastructure (COMPLETED)

**Components:**
- `ALFWorldEnvWrapper`: Clean interface to ALFWorld environments
- `ALFWorldKernelAdapter`: Core adapter bridging ALFWorld and Kernel System
- `state_bridge.py`: State conversion utilities
  - `obs_to_kernel_state()`: Convert ALFWorld observation to Kernel State
  - `update_kernel_state()`: Update Kernel State with new observation
  - `extract_action()`: Extract action from Kernel State for ALFWorld

**Tests:** 11/11 passing
- `test_env_wrapper.py`: Environment wrapper tests
- `test_adapter.py`: Adapter initialization tests
- `test_state_bridge.py`: State conversion tests

**Key Design Decisions:**
1. **Direct Text Observation**: ALFWorld text observations passed directly to Architect via `user_prompt`, leveraging LLM's natural language understanding
2. **ActionWorker Direct Output**: Worker outputs `selected_action` field directly with executable ALFWorld commands
3. **Simple State Structure**: `domain_state` uses simple string fields instead of complex structured data

### 🔄 Phase 2: Core Adapter (Next)
- Implement full `run_episode()` flow
- Test adapter with ALFWorld environment
- Verify state conversion and action extraction

### ⏳ Phase 3: Workers Implementation
- Implement `ActionWorker`
- Implement `ALFWorldArchitect`
- Register workers to Kernel System

### ⏳ Phase 4: End-to-End Integration
- Implement evaluators
- Create example scripts
- Run complete evaluation

## Running Tests

```bash
# Activate conda environment
conda activate blackboard

# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_state_bridge.py -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

## Key Concepts

### domain_state
Business state container in Kernel System that stores all task execution data. For ALFWorld tasks:

```python
domain_state = {
    "task_goal": "Put a clean apple in the fridge",
    "current_observation": "You are in the kitchen. You see a apple 1 on the table 1.",
    "available_actions": ["go to table 1", "go to fridge 1", "take apple 1", "look"],
    "selected_action": "",  # ActionWorker fills this
    "action_history": [],
    "observation_history": [],
    "status": "planning"
}
```

### Execution Flow

1. **ALFWorld Reset** → Initial observation
2. **obs_to_kernel_state()** → Convert to Kernel State with user_prompt
3. **Kernel Graph Invoke** → Architect analyzes, ActionWorker selects action
4. **extract_action()** → Read `selected_action` from domain_state
5. **ALFWorld Step** → Execute action, get new observation
6. **update_kernel_state()** → Inject new observation, update history
7. **Repeat** until task complete or max_steps

## Dependencies

- `alfworld>=0.3.3`: ALFWorld environment
- `pyyaml>=6.0`: Configuration management
- `pytest>=7.0.0`: Testing framework
- LangGraph Kernel System (imported from parent project)

## Next Steps

1. ✅ Phase 1 completed with all tests passing
2. Proceed to Phase 2: Implement full adapter integration
3. Test with actual ALFWorld environment
4. Move to Phase 3: Implement Workers

## References

- Implementation Plan: `IMPLEMENTATION_PLAN.md`
- LangGraph Kernel System: `/home/syq/Documents/blackboard/langgraph/libs/kernel_system/`
- ALFWorld: https://github.com/alfworld/alfworld
