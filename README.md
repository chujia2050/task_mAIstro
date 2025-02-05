# Task mAIstro

Managing tasks effectively is a universal challenge. Task mAIstro is an AI-powered task management agent that combines natural language processing with long-term memory to create a more intuitive and adaptive experience. 

Key features:
* Natural conversation through text to update or add tasks
* Adaptive learning of your management style and preferences
* Persistent memory of tasks, context, and preferences

## Task mAIstro Application

### Architecture

Task mAIstro leverages [LangGraph](https://langchain-ai.github.io/langgraph/) to maintain three memory types:

1. **ToDo List Memory**
   - Task descriptions and deadlines
   - Time estimates and status tracking
   - Actionable next steps

2. **User Profile Memory**
   - Personal preferences and context
   - Work/life patterns
   - Historical interactions

3. **Interaction Memory**
   - Task management style
   - Communication preferences
   - Organizational patterns

The schema for each memory type as well as the graph flow is defined in `task_maistro.py`. 
The graph flow is orchestrated by a central `task_maistro` node that:
- Chooses to update one of the three memories based on the user's input
- Uses tool calling with the [Trustcall library](https://github.com/hinthornw/trustcall) to update the chosen memory type
