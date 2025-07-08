import { Agent, Task, LLMModel, OrchestratorDecision, ConversationContext } from '../types/agents'

export class OrchestratorService {
  private agents: Agent[] = []
  private models: LLMModel[] = []
  private conversationHistory: ConversationContext[] = []
  private logger: (level: string, message: string, data?: any) => void

  constructor(logger?: (level: string, message: string, data?: any) => void) {
    this.logger = logger || console.log
    this.initializeAgents()
    this.initializeModels()
  }

  private initializeAgents(): void {
    this.agents = [
      {
        id: 'orchestrator',
        name: 'Task Orchestrator',
        type: 'orchestrator',
        model: 'llama3.2:3b',
        capabilities: ['task_analysis', 'agent_selection', 'coordination'],
        status: 'idle',
        performance: { avgResponseTime: 0.8, successRate: 0.98, tasksCompleted: 0 },
        cost: { tokensPerSecond: 100, costPerToken: 0.0001 },
        specializations: ['meta_reasoning', 'decision_making']
      },
      {
        id: 'chat-assistant',
        name: 'General Chat Assistant',
        type: 'chat',
        model: 'llama3.2:3b',
        capabilities: ['conversation', 'general_qa', 'task_coordination', 'summarization'],
        status: 'idle',
        performance: { avgResponseTime: 1.2, successRate: 0.95, tasksCompleted: 0 },
        cost: { tokensPerSecond: 80, costPerToken: 0.0001 },
        specializations: ['general_knowledge', 'conversation', 'help']
      },
      {
        id: 'code-specialist',
        name: 'Code Specialist',
        type: 'coding',
        model: 'qwen2.5-coder:32b',
        capabilities: ['code_generation', 'debugging', 'code_review', 'architecture', 'documentation'],
        status: 'idle',
        performance: { avgResponseTime: 3.8, successRate: 0.92, tasksCompleted: 0 },
        cost: { tokensPerSecond: 40, costPerToken: 0.0003 },
        specializations: ['programming', 'software_engineering', 'algorithms']
      },
      {
        id: 'vision-analyst',
        name: 'Vision Analyst',
        type: 'vision',
        model: 'llava:34b',
        capabilities: ['image_analysis', 'visual_qa', 'diagram_reading', 'ocr'],
        status: 'idle',
        performance: { avgResponseTime: 5.2, successRate: 0.88, tasksCompleted: 0 },
        cost: { tokensPerSecond: 20, costPerToken: 0.0005 },
        specializations: ['computer_vision', 'image_understanding', 'visual_reasoning']
      },
      {
        id: 'reasoning-engine',
        name: 'Logic & Reasoning Engine',
        type: 'reasoning',
        model: 'deepseek-r1:7b',
        capabilities: ['complex_reasoning', 'problem_solving', 'mathematics', 'logic'],
        status: 'idle',
        performance: { avgResponseTime: 4.1, successRate: 0.91, tasksCompleted: 0 },
        cost: { tokensPerSecond: 50, costPerToken: 0.0002 },
        specializations: ['mathematical_reasoning', 'logical_analysis', 'problem_solving']
      },
      {
        id: 'research-assistant',
        name: 'Research Assistant',
        type: 'research',
        model: 'llama3.3:70b-instruct-q4_K_M',
        capabilities: ['research', 'analysis', 'long_form_content', 'synthesis'],
        status: 'idle',
        performance: { avgResponseTime: 8.5, successRate: 0.94, tasksCompleted: 0 },
        cost: { tokensPerSecond: 15, costPerToken: 0.0008 },
        specializations: ['research', 'analysis', 'comprehensive_responses']
      },
      {
        id: 'memory-manager',
        name: 'Memory Manager',
        type: 'memory',
        model: 'llama3.2:3b',
        capabilities: ['memory_retrieval', 'context_management', 'information_synthesis'],
        status: 'idle',
        performance: { avgResponseTime: 1.5, successRate: 0.96, tasksCompleted: 0 },
        cost: { tokensPerSecond: 70, costPerToken: 0.0001 },
        specializations: ['memory_operations', 'context_retrieval', 'information_management']
      }
    ]
  }

  private initializeModels(): void {
    this.models = [
      {
        id: 'llama3.2:1b',
        name: 'Llama 3.2 1B',
        provider: 'ollama',
        capabilities: ['basic_tasks', 'simple_qa', 'quick_responses'],
        contextWindow: 4096,
        speed: 'very_fast',
        cost: 0.5,
        quality: 6,
        specialties: ['speed', 'efficiency'],
        availability: 'available',
        currentLoad: 0
      },
      {
        id: 'llama3.2:3b',
        name: 'Llama 3.2 3B',
        provider: 'ollama',
        capabilities: ['balanced_tasks', 'general_qa', 'moderate_reasoning'],
        contextWindow: 4096,
        speed: 'fast',
        cost: 1,
        quality: 7.5,
        specialties: ['balance', 'general_purpose'],
        availability: 'available',
        currentLoad: 0
      },
      {
        id: 'qwen2.5-coder:32b',
        name: 'Qwen 2.5 Coder 32B',
        provider: 'ollama',
        capabilities: ['coding', 'debugging', 'code_review', 'architecture'],
        contextWindow: 32768,
        speed: 'medium',
        cost: 3,
        quality: 9,
        specialties: ['programming', 'code_generation'],
        availability: 'available',
        currentLoad: 0
      },
      {
        id: 'llava:34b',
        name: 'LLaVA 34B',
        provider: 'ollama',
        capabilities: ['vision', 'image_understanding', 'visual_qa'],
        contextWindow: 4096,
        speed: 'slow',
        cost: 4,
        quality: 8.5,
        specialties: ['vision', 'multimodal'],
        availability: 'available',
        currentLoad: 0
      },
      {
        id: 'deepseek-r1:7b',
        name: 'DeepSeek R1 7B',
        provider: 'ollama',
        capabilities: ['reasoning', 'problem_solving', 'mathematics'],
        contextWindow: 8192,
        speed: 'fast',
        cost: 2,
        quality: 8.8,
        specialties: ['reasoning', 'mathematics'],
        availability: 'available',
        currentLoad: 0
      },
      {
        id: 'llama3.3:70b-instruct-q4_K_M',
        name: 'Llama 3.3 70B Instruct',
        provider: 'ollama',
        capabilities: ['advanced_reasoning', 'research', 'complex_analysis'],
        contextWindow: 8192,
        speed: 'slow',
        cost: 5,
        quality: 9.5,
        specialties: ['research', 'complex_reasoning'],
        availability: 'available',
        currentLoad: 0
      }
    ]
  }

  public async analyzeTask(userInput: string, context?: ConversationContext): Promise<Task> {
    this.logger('info', 'Analyzing user task', { input: userInput })

    // Analyze task complexity and requirements
    const task: Task = {
      id: `task_${Date.now()}`,
      type: this.determineTaskType(userInput),
      content: userInput,
      priority: this.calculatePriority(userInput),
      complexity: this.assessComplexity(userInput),
      requiresVision: this.requiresVision(userInput),
      requiresMemory: this.requiresMemory(userInput, context),
      estimatedTokens: this.estimateTokens(userInput),
      maxResponseTime: this.determineMaxResponseTime(userInput),
      status: 'analyzing'
    }

    this.logger('info', 'Task analysis complete', task)
    return task
  }

  public async selectOptimalAgentAndModel(task: Task, context?: ConversationContext): Promise<OrchestratorDecision> {
    this.logger('info', 'Selecting optimal agent and model', { taskId: task.id })

    // Filter suitable agents
    const suitableAgents = this.agents.filter(agent => 
      this.isAgentSuitable(agent, task) && agent.status === 'idle'
    )

    if (suitableAgents.length === 0) {
      throw new Error('No suitable agents available')
    }

    // Score agents based on multiple criteria
    const agentScores = suitableAgents.map(agent => ({
      agent,
      score: this.scoreAgent(agent, task, context)
    }))

    // Sort by score (highest first)
    agentScores.sort((a, b) => b.score - a.score)
    const selectedAgent = agentScores[0].agent

    // Select optimal model for the chosen agent
    const suitableModels = this.models.filter(model => 
      this.isModelSuitable(model, task, selectedAgent)
    )

    const modelScores = suitableModels.map(model => ({
      model,
      score: this.scoreModel(model, task, selectedAgent)
    }))

    modelScores.sort((a, b) => b.score - a.score)
    const selectedModel = modelScores[0].model

    // Generate reasoning
    const reasoning = this.generateSelectionReasoning(selectedAgent, selectedModel, task)

    // Calculate estimates
    const estimatedCost = this.calculateEstimatedCost(task, selectedModel)
    const estimatedTime = this.calculateEstimatedTime(task, selectedAgent, selectedModel)

    // Prepare fallback options
    const fallbackOptions = agentScores.slice(1, 3).map(({ agent }) => {
      const fallbackModel = this.selectBestModelForAgent(agent, task)
      return {
        agent,
        model: fallbackModel,
        reason: `Fallback option with ${agent.name} using ${fallbackModel.name}`
      }
    })

    const decision: OrchestratorDecision = {
      selectedAgent,
      selectedModel,
      reasoning,
      confidence: this.calculateConfidence(selectedAgent, selectedModel, task),
      estimatedCost,
      estimatedTime,
      fallbackOptions
    }

    this.logger('info', 'Agent and model selection complete', decision)
    return decision
  }

  private determineTaskType(input: string): string {
    const lowerInput = input.toLowerCase()
    
    if (lowerInput.includes('code') || lowerInput.includes('program') || lowerInput.includes('debug')) {
      return 'coding'
    }
    if (lowerInput.includes('image') || lowerInput.includes('picture') || lowerInput.includes('visual')) {
      return 'vision'
    }
    if (lowerInput.includes('math') || lowerInput.includes('calculate') || lowerInput.includes('solve')) {
      return 'reasoning'
    }
    if (lowerInput.includes('research') || lowerInput.includes('analyze') || lowerInput.includes('study')) {
      return 'research'
    }
    if (lowerInput.includes('remember') || lowerInput.includes('recall') || lowerInput.includes('previous')) {
      return 'memory'
    }
    
    return 'chat'
  }

  private calculatePriority(input: string): number {
    const urgentKeywords = ['urgent', 'asap', 'immediately', 'critical', 'emergency']
    const highKeywords = ['important', 'priority', 'soon', 'quickly']
    
    const lowerInput = input.toLowerCase()
    
    if (urgentKeywords.some(keyword => lowerInput.includes(keyword))) return 9
    if (highKeywords.some(keyword => lowerInput.includes(keyword))) return 7
    
    return 5 // Default priority
  }

  private assessComplexity(input: string): 'simple' | 'medium' | 'complex' {
    const complexKeywords = ['complex', 'detailed', 'comprehensive', 'analyze', 'research']
    const simpleKeywords = ['simple', 'quick', 'basic', 'brief']
    
    const lowerInput = input.toLowerCase()
    const wordCount = input.split(' ').length
    
    if (complexKeywords.some(keyword => lowerInput.includes(keyword)) || wordCount > 50) {
      return 'complex'
    }
    if (simpleKeywords.some(keyword => lowerInput.includes(keyword)) || wordCount < 10) {
      return 'simple'
    }
    
    return 'medium'
  }

  private requiresVision(input: string): boolean {
    const visionKeywords = ['image', 'picture', 'photo', 'visual', 'diagram', 'chart', 'graph', 'screenshot']
    return visionKeywords.some(keyword => input.toLowerCase().includes(keyword))
  }

  private requiresMemory(input: string, context?: ConversationContext): boolean {
    const memoryKeywords = ['remember', 'recall', 'previous', 'before', 'earlier', 'last time']
    return memoryKeywords.some(keyword => input.toLowerCase().includes(keyword)) || 
           (context && context.messages.length > 5)
  }

  private estimateTokens(input: string): number {
    // Rough estimation: 1 token ≈ 0.75 words
    const wordCount = input.split(' ').length
    return Math.ceil(wordCount / 0.75)
  }

  private determineMaxResponseTime(input: string): number {
    const urgentKeywords = ['urgent', 'quick', 'fast', 'immediately']
    const lowerInput = input.toLowerCase()
    
    if (urgentKeywords.some(keyword => lowerInput.includes(keyword))) {
      return 5000 // 5 seconds
    }
    
    return 30000 // 30 seconds default
  }

  private isAgentSuitable(agent: Agent, task: Task): boolean {
    // Check if agent type matches task requirements
    if (task.requiresVision && agent.type !== 'vision') return false
    if (task.type === 'coding' && agent.type !== 'coding') return false
    if (task.type === 'reasoning' && agent.type !== 'reasoning') return false
    if (task.type === 'research' && agent.type !== 'research') return false
    if (task.requiresMemory && agent.type !== 'memory') return false
    
    // Check if agent has required capabilities
    const requiredCapabilities = this.getRequiredCapabilities(task)
    return requiredCapabilities.every(cap => agent.capabilities.includes(cap))
  }

  private isModelSuitable(model: LLMModel, task: Task, agent: Agent): boolean {
    // Check if model is available
    if (model.availability !== 'available') return false
    
    // Check if model supports required capabilities
    const requiredCapabilities = this.getRequiredCapabilities(task)
    const hasRequiredCaps = requiredCapabilities.some(cap => model.capabilities.includes(cap))
    
    // Check context window requirements
    const contextRequired = task.estimatedTokens * 2 // Input + output
    if (model.contextWindow < contextRequired) return false
    
    return hasRequiredCaps
  }

  private scoreAgent(agent: Agent, task: Task, context?: ConversationContext): number {
    let score = 0
    
    // Performance metrics (40% weight)
    score += agent.performance.successRate * 40
    score += (1 / agent.performance.avgResponseTime) * 10 // Faster is better
    
    // Specialization match (30% weight)
    const taskSpecialties = this.getTaskSpecialties(task)
    const matchingSpecialties = agent.specializations.filter(spec => 
      taskSpecialties.includes(spec)
    ).length
    score += (matchingSpecialties / Math.max(taskSpecialties.length, 1)) * 30
    
    // Recent usage (20% weight) - prefer less recently used agents for load balancing
    const timeSinceLastUse = agent.performance.lastUsed ? 
      Date.now() - new Date(agent.performance.lastUsed).getTime() : 
      Infinity
    score += Math.min(timeSinceLastUse / (1000 * 60 * 60), 20) // Max 20 points for 1+ hour
    
    // Task complexity match (10% weight)
    if (task.complexity === 'complex' && agent.type === 'research') score += 10
    if (task.complexity === 'simple' && agent.type === 'chat') score += 10
    
    return score
  }

  private scoreModel(model: LLMModel, task: Task, agent: Agent): number {
    let score = 0
    
    // Quality vs Cost balance (40% weight)
    const qualityCostRatio = model.quality / model.cost
    score += qualityCostRatio * 10
    
    // Speed requirements (30% weight)
    const speedScore = this.getSpeedScore(model.speed, task.maxResponseTime)
    score += speedScore * 30
    
    // Capability match (20% weight)
    const requiredCaps = this.getRequiredCapabilities(task)
    const matchingCaps = model.capabilities.filter(cap => requiredCaps.includes(cap)).length
    score += (matchingCaps / Math.max(requiredCaps.length, 1)) * 20
    
    // Current load (10% weight)
    score += (100 - model.currentLoad) / 10
    
    return score
  }

  private getRequiredCapabilities(task: Task): string[] {
    const capabilities = []
    
    if (task.requiresVision) capabilities.push('vision', 'image_understanding')
    if (task.type === 'coding') capabilities.push('coding', 'code_generation')
    if (task.type === 'reasoning') capabilities.push('reasoning', 'problem_solving')
    if (task.type === 'research') capabilities.push('research', 'analysis')
    if (task.requiresMemory) capabilities.push('memory_retrieval')
    
    return capabilities
  }

  private getTaskSpecialties(task: Task): string[] {
    const specialties = []
    
    if (task.type === 'coding') specialties.push('programming', 'software_engineering')
    if (task.type === 'vision') specialties.push('computer_vision', 'image_understanding')
    if (task.type === 'reasoning') specialties.push('mathematical_reasoning', 'logical_analysis')
    if (task.type === 'research') specialties.push('research', 'analysis')
    if (task.complexity === 'complex') specialties.push('complex_reasoning')
    
    return specialties
  }

  private getSpeedScore(modelSpeed: string, maxResponseTime: number): number {
    const speedValues = {
      'very_fast': 10,
      'fast': 7,
      'medium': 5,
      'slow': 2
    }
    
    const baseScore = speedValues[modelSpeed] || 5
    
    // Boost score if task requires fast response
    if (maxResponseTime <= 5000 && modelSpeed === 'very_fast') return baseScore * 1.5
    if (maxResponseTime <= 10000 && ['very_fast', 'fast'].includes(modelSpeed)) return baseScore * 1.2
    
    return baseScore
  }

  private selectBestModelForAgent(agent: Agent, task: Task): LLMModel {
    const suitableModels = this.models.filter(model => 
      this.isModelSuitable(model, task, agent)
    )
    
    if (suitableModels.length === 0) {
      return this.models[0] // Fallback to first available model
    }
    
    const modelScores = suitableModels.map(model => ({
      model,
      score: this.scoreModel(model, task, agent)
    }))
    
    modelScores.sort((a, b) => b.score - a.score)
    return modelScores[0].model
  }

  private generateSelectionReasoning(agent: Agent, model: LLMModel, task: Task): string {
    const reasons = []
    
    reasons.push(`Selected ${agent.name} for its specialization in ${agent.specializations.join(', ')}`)
    reasons.push(`Chosen ${model.name} for optimal balance of quality (${model.quality}/10) and cost (${model.cost})`)
    
    if (task.complexity === 'complex') {
      reasons.push('Task complexity requires advanced reasoning capabilities')
    }
    
    if (task.requiresVision) {
      reasons.push('Task requires visual processing capabilities')
    }
    
    if (task.maxResponseTime <= 10000) {
      reasons.push('Fast response time required, prioritizing speed')
    }
    
    return reasons.join('. ')
  }

  private calculateEstimatedCost(task: Task, model: LLMModel): number {
    const estimatedOutputTokens = task.estimatedTokens * 2 // Assume 2x input for output
    const totalTokens = task.estimatedTokens + estimatedOutputTokens
    return totalTokens * model.cost * 0.001 // Convert to reasonable cost units
  }

  private calculateEstimatedTime(task: Task, agent: Agent, model: LLMModel): number {
    const baseTime = agent.performance.avgResponseTime * 1000 // Convert to ms
    const complexityMultiplier = {
      'simple': 0.7,
      'medium': 1.0,
      'complex': 1.5
    }[task.complexity]
    
    const speedMultiplier = {
      'very_fast': 0.5,
      'fast': 0.8,
      'medium': 1.0,
      'slow': 1.5
    }[model.speed]
    
    return baseTime * complexityMultiplier * speedMultiplier
  }

  private calculateConfidence(agent: Agent, model: LLMModel, task: Task): number {
    let confidence = 0.5 // Base confidence
    
    // Agent performance history
    confidence += agent.performance.successRate * 0.3
    
    // Model quality
    confidence += (model.quality / 10) * 0.2
    
    // Capability match
    const requiredCaps = this.getRequiredCapabilities(task)
    const agentCapMatch = agent.capabilities.filter(cap => requiredCaps.includes(cap)).length / requiredCaps.length
    const modelCapMatch = model.capabilities.filter(cap => requiredCaps.includes(cap)).length / requiredCaps.length
    confidence += (agentCapMatch + modelCapMatch) * 0.25
    
    return Math.min(confidence, 1.0)
  }

  public updateAgentPerformance(agentId: string, success: boolean, responseTime: number): void {
    const agent = this.agents.find(a => a.id === agentId)
    if (!agent) return
    
    // Update performance metrics with exponential moving average
    agent.performance.avgResponseTime = agent.performance.avgResponseTime * 0.9 + responseTime * 0.1
    agent.performance.successRate = agent.performance.successRate * 0.95 + (success ? 1 : 0) * 0.05
    agent.performance.tasksCompleted += 1
    agent.performance.lastUsed = new Date().toISOString()
    
    this.logger('info', 'Updated agent performance', { agentId, performance: agent.performance })
  }

  public updateModelLoad(modelId: string, loadChange: number): void {
    const model = this.models.find(m => m.id === modelId)
    if (!model) return
    
    model.currentLoad = Math.max(0, Math.min(100, model.currentLoad + loadChange))
    
    this.logger('info', 'Updated model load', { modelId, currentLoad: model.currentLoad })
  }

  public getAgents(): Agent[] {
    return [...this.agents]
  }

  public getModels(): LLMModel[] {
    return [...this.models]
  }

  public getAgentById(id: string): Agent | undefined {
    return this.agents.find(agent => agent.id === id)
  }

  public getModelById(id: string): LLMModel | undefined {
    return this.models.find(model => model.id === id)
  }
}