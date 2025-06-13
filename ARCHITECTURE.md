expert_reasoning_system/
├── .env                  # Environment variable configuration file
├── main.py               # Main entry file
├── requirements.txt      # Project dependencies
├── README.md             # Project documentation
├── agent/
│   ├── __init__.py
│   ├── agent.py          # Large model invocation logic
│   ├── evaluator.py      # Answer evaluation logic
│   └── router.py         # Question routing logic
├── tools/
│   ├── __init__.py
│   ├── search.py         # Search tool
│   ├── paper.py          # Paper processing tool
│   ├── code_executor.py  # Code execution tool
│   └── utils.py          # Utility functions
└── memory/
    ├── __init__.py
    └── storage.py        # Result storage