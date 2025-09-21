# Prompt Optimizer - System Architecture

## Overview Diagram

```mermaid
graph TB
    subgraph "User Interface Layer"
        A[User Application] --> B[PromptOptimizer API]
    end
    
    subgraph "Core Logic Layer"
        B --> C[Model Registry]
        B --> D[Family Detection]
        B --> E[Prompt Manager]
        B --> F[Result Formatter]
    end
    
    subgraph "External Services"
        E --> G[OpenRouter API]
    end
    
    subgraph "Data Layer"
        C --> H[Model Family Mappings]
        E --> I[Family-Specific Prompts]
    end
    
    subgraph "Supporting Components"
        J[Error Handler] --> B
        K[Rate Limiter] --> G
        L[Logger] --> B
        M[Config Manager] --> B
    end
    
    A -->|optimize()| B
    B -->|detect_family()| D
    D -->|get_family()| C
    C -->|get_mappings()| H
    B -->|get_prompt()| E
    E -->|get_family_prompt()| I
    B -->|call_api()| G
    G -->|response| F
    F -->|result| A
```

## Data Flow Architecture

```mermaid
sequenceDiagram
    participant U as User
    participant O as PromptOptimizer
    participant R as Model Registry
    participant P as Prompt Manager
    participant C as OpenRouter Client
    participant API as OpenRouter API
    
    U->>O: optimize(prompt, target_model)
    O->>R: detect_model_family(target_model)
    R-->>O: ModelFamily
    O->>P: get_optimization_prompt(ModelFamily)
    P-->>O: System Prompt
    O->>C: optimize_prompt(original_prompt, system_prompt)
    C->>API: POST /chat/completions
    API-->>C: Optimized Response
    C-->>O: Optimized Prompt
    O->>O: format_result()
    O-->>U: OptimizationResult
```

## Component Breakdown

### 1. Model Registry
```mermaid
graph LR
    subgraph "Model Registry"
        A[ModelFamily Enum] --> B[OpenAI]
        A --> C[Anthropic]
        A --> D[Google]
        A --> E[Meta]
        A --> F[Mistral]
        
        B --> G[gpt-5]
        B --> H[gpt-4.1]
        B --> I[gpt-4o]
        B --> J[gpt-4-turbo]
        B --> K[gpt-3.5-turbo]
        
        C --> L[claude-4]
        C --> M[claude-4.1]
        C --> N[claude-3.5-sonnet]
        C --> O[claude-3-opus]
        C --> P[claude-3-haiku]
        
        D --> Q[gemini-2.5-pro]
        D --> R[gemini-2.5-flash]
        D --> S[gemini-1.5-pro]
        D --> T[gemini-1.5-flash]
        
        E --> U[llama-3.1-70b]
        E --> V[llama-3.1-8b]
        E --> W[llama-3-70b]
        E --> X[llama-3-8b]
        
        F --> Y[mixtral-8x7b]
        F --> Z[mistral-7b-instruct]
    end
```

### 2. Optimization Prompt System
```mermaid
graph TB
    subgraph "Optimization Prompts"
        A[Base Template] --> B[OpenAI Prompts]
        A --> C[Anthropic Prompts]
        A --> D[Google Prompts]
        A --> E[Meta Prompts]
        A --> F[Mistral Prompts]
        
        B --> G[GPT-5 Specific]
        B --> H[GPT-4.1 Specific]
        B --> I[GPT-4o Specific]
        
        C --> J[Claude 4 Specific]
        C --> K[Claude 4.1 Specific]
        C --> L[Claude 3.5 Specific]
        
        D --> M[Gemini 2.5 Specific]
        D --> N[Gemini 1.5 Specific]
        
        E --> O[Llama 3.1 Specific]
        E --> P[Llama 3 Specific]
        
        F --> Q[Mixtral Specific]
        F --> R[Mistral 7B Specific]
    end
```

### 3. Error Handling Flow
```mermaid
graph TD
    A[API Call] --> B{Success?}
    B -->|Yes| C[Return Result]
    B -->|No| D{Error Type}
    D -->|Network| E[Retry with Backoff]
    D -->|Rate Limit| F[Wait and Retry]
    D -->|Auth| G[Throw AuthError]
    D -->|Invalid| H[Throw ValidationError]
    D -->|Timeout| I[Retry with Timeout]
    E --> A
    F --> A
    I --> A
```

## Performance Architecture

```mermaid
graph LR
    subgraph "Performance Optimization"
        A[Connection Pooling] --> B[HTTP Client]
        C[Prompt Caching] --> D[Optimization Prompts]
        E[Rate Limiting] --> F[API Calls]
        G[Async Processing] --> H[Non-blocking I/O]
        I[Memory Management] --> J[Efficient String Handling]
    end
    
    B --> K[Faster API Calls]
    D --> L[Reduced Prompt Generation]
    F --> M[Prevent API Abuse]
    H --> N[Better Throughput]
    J --> O[Lower Memory Usage]
    
    K --> P[< 10s Optimization]
    L --> P
    M --> P
    N --> P
    O --> P
```

## Deployment Architecture

```mermaid
graph TB
    subgraph "Development"
        A[Local Testing] --> B[uv run]
        B --> C[pytest]
        C --> D[Coverage Reports]
    end
    
    subgraph "Package Distribution"
        E[pyproject.toml] --> F[Build Package]
        F --> G[Upload to PyPI]
        G --> H[Pip Installation]
    end
    
    subgraph "Environment Setup"
        I[.env.example] --> J[Environment Variables]
        J --> K[Configuration Management]
        K --> L[Runtime Settings]
    end
    
    subgraph "Documentation"
        M[README.md] --> N[API Reference]
        N --> O[Usage Examples]
        O --> P[Troubleshooting Guide]
    end
```

This architecture provides a comprehensive view of how the Prompt Optimizer system will be structured and how data flows through the various components.