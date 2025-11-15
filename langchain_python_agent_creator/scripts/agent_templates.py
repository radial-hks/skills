#!/usr/bin/env python3
"""
Agent Template Generator

This script generates ready-to-use agent templates for common use cases,
including customer support, data analysis, code review, and more.
"""

import os
from typing import List, Dict, Any, Optional
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.tools import BaseTool

class AgentTemplateGenerator:
    """Generate pre-configured agent templates for common use cases."""
    
    def __init__(self):
        """Initialize the template generator."""
        self.templates = {}
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize built-in templates."""
        self.templates = {
            "customer_support": self.create_customer_support_agent,
            "data_analyst": self.create_data_analyst_agent,
            "code_reviewer": self.create_code_reviewer_agent,
            "content_writer": self.create_content_writer_agent,
            "research_assistant": self.create_research_assistant_agent,
            "math_tutor": self.create_math_tutor_agent,
            "language_translator": self.create_language_translator_agent,
            "file_processor": self.create_file_processor_agent
        }
    
    def create_customer_support_agent(self, model: str = "claude-3-sonnet-20240229") -> Any:
        """
        Create a customer support agent template.
        
        Args:
            model: Model to use for the agent
            
        Returns:
            Configured customer support agent
        """
        
        @tool
        def search_knowledge_base(query: str) -> str:
            """Search the knowledge base for relevant information."""
            return f"Knowledge base results for '{query}': Found 3 relevant articles about account management, billing, and technical issues."
        
        @tool
        def create_support_ticket(customer_issue: str, priority: str = "medium") -> str:
            """Create a support ticket for customer issues."""
            ticket_id = f"TK-{hash(customer_issue) % 10000:04d}"
            return f"Support ticket created: {ticket_id} (Priority: {priority}). Our team will respond within 24 hours."
        
        @tool
        def escalate_to_human(customer_issue: str, escalation_reason: str) -> str:
            """Escalate complex issues to human support."""
            return f"Issue escalated to human support team. Reason: {escalation_reason}. A specialist will contact you within 2 hours."
        
        @tool
        def check_account_status(customer_id: str) -> str:
            """Check customer account status and details."""
            return f"Account {customer_id} status: Active, Premium plan, Last login: 2 hours ago, No outstanding issues."
        
        system_prompt = """You are a helpful customer support assistant. Your goals are:
        1. Provide accurate and helpful information
        2. Resolve customer issues efficiently
        3. Escalate complex issues when necessary
        4. Maintain a friendly and professional tone
        5. Search the knowledge base for relevant information
        
        Always be empathetic and solution-oriented. If you cannot resolve an issue, create a ticket or escalate to human support."""
        
        return create_agent(
            model=model,
            tools=[search_knowledge_base, create_support_ticket, escalate_to_human, check_account_status],
            system_prompt=system_prompt
        )
    
    def create_data_analyst_agent(self, model: str = "claude-3-sonnet-20240229") -> Any:
        """
        Create a data analysis agent template.
        
        Args:
            model: Model to use for the agent
            
        Returns:
            Configured data analyst agent
        """
        
        @tool
        def load_dataset(dataset_name: str) -> str:
            """Load a dataset for analysis."""
            return f"Dataset '{dataset_name}' loaded successfully. Contains 1000 rows and 15 columns. Data types: numeric(8), categorical(4), datetime(3)."
        
        @tool
        def generate_statistics(column_name: str, dataset_name: str) -> str:
            """Generate statistical summary for a dataset column."""
            return f"Statistics for {column_name} in {dataset_name}: Mean=25.3, Median=24.8, Std=4.2, Min=12.1, Max=38.7, Missing values: 2%"
        
        @tool
        def create_visualization(chart_type: str, column_name: str, dataset_name: str) -> str:
            """Create data visualizations."""
            return f"Created {chart_type} visualization for {column_name} in {dataset_name}. Chart shows normal distribution with slight right skew."
        
        @tool
        def detect_anomalies(column_name: str, dataset_name: str) -> str:
            """Detect anomalies and outliers in data."""
            return f"Anomaly detection for {column_name}: Found 3 outliers (values > 2 std dev), 2 data quality issues detected."
        
        @tool
        def suggest_analysis(column_name: str, dataset_name: str) -> str:
            """Suggest appropriate analysis techniques."""
            return f"Suggested analyses for {column_name}: Correlation analysis, Trend analysis, Seasonal decomposition. Recommended visualization: Line chart with confidence intervals."
        
        system_prompt = """You are a professional data analyst. Your responsibilities:
        1. Load and validate datasets
        2. Generate comprehensive statistical summaries
        3. Create appropriate visualizations
        4. Detect anomalies and data quality issues
        5. Suggest relevant analysis techniques
        6. Provide actionable insights and recommendations
        
        Always explain your analysis approach and provide clear, actionable insights. Focus on statistical accuracy and business relevance."""
        
        return create_agent(
            model=model,
            tools=[load_dataset, generate_statistics, create_visualization, detect_anomalies, suggest_analysis],
            system_prompt=system_prompt
        )
    
    def create_code_reviewer_agent(self, model: str = "claude-3-sonnet-20240229") -> Any:
        """
        Create a code review agent template.
        
        Args:
            model: Model to use for the agent
            
        Returns:
            Configured code review agent
        """
        
        @tool
        def analyze_code_quality(code: str) -> str:
            """Analyze code quality and structure."""
            return "Code quality analysis: Clean structure, good variable naming, proper indentation. Suggestions: Add more comments, consider extracting complex logic into separate functions."
        
        @tool
        def check_security_issues(code: str) -> str:
            """Check for security vulnerabilities."""
            return "Security analysis: No SQL injection risks detected, input validation present. Warning: Hardcoded API key found on line 15. Recommendation: Use environment variables."
        
        @tool
        def suggest_performance_improvements(code: str) -> str:
            """Suggest performance optimizations."""
            return "Performance suggestions: Use list comprehension instead of loop on line 23, consider caching for repeated calculations, database query could be optimized with indexing."
        
        @tool
        def check_code_style(code: str) -> str:
            """Check code style and conventions."""
            return "Style check: Follows PEP 8 conventions, good function naming. Issues: Line 45 exceeds 80 characters, missing docstring for function on line 12."
        
        @tool
        def run_tests(code: str) -> str:
            """Simulate running tests on code."""
            return "Test results: 8/10 tests passing. Failed tests: test_edge_case_empty_input, test_performance_large_dataset. Coverage: 85% of code covered."
        
        system_prompt = """You are an experienced code reviewer. Your responsibilities:
        1. Analyze code quality and structure
        2. Identify security vulnerabilities
        3. Suggest performance improvements
        4. Check adherence to coding standards
        5. Review test coverage and quality
        
        Provide constructive feedback with specific line references and actionable suggestions. Focus on maintainability, security, and performance."""
        
        return create_agent(
            model=model,
            tools=[analyze_code_quality, check_security_issues, suggest_performance_improvements, check_code_style, run_tests],
            system_prompt=system_prompt
        )
    
    def create_content_writer_agent(self, model: str = "claude-3-sonnet-20240229") -> Any:
        """
        Create a content writing agent template.
        
        Args:
            model: Model to use for the agent
            
        Returns:
            Configured content writer agent
        """
        
        @tool
        def research_topic(topic: str) -> str:
            """Research information about a topic."""
            return f"Research findings for '{topic}': Key points identified, 5 authoritative sources found, main themes outlined, relevant statistics gathered."
        
        @tool
        def generate_outline(topic: str, content_type: str) -> str:
            """Generate content outline and structure."""
            return f"Generated {content_type} outline for '{topic}': Introduction, 3 main sections with subsections, conclusion, call-to-action. Estimated length: 800-1000 words."
        
        @tool
        def optimize_seo(content: str, keywords: List[str]) -> str:
            """Optimize content for search engines."""
            return f"SEO optimization applied: Keywords {keywords} integrated naturally, meta description created, heading structure optimized, internal linking suggestions provided."
        
        @tool
        def check_readability(content: str) -> str:
            """Check content readability and clarity."""
            return "Readability analysis: Flesch score 65 (good), average sentence length 15 words, 2 passive voice constructions found. Suggestions: Break up 3 long sentences, use more active voice."
        
        @tool
        def suggest_improvements(content: str) -> str:
            """Suggest content improvements."""
            return "Content suggestions: Add more specific examples, include recent statistics, strengthen opening hook, add compelling call-to-action, consider adding visual elements."
        
        system_prompt = """You are a professional content writer and editor. Your capabilities:
        1. Research topics thoroughly
        2. Create well-structured content outlines
        3. Write engaging and informative content
        4. Optimize content for SEO
        5. Ensure readability and clarity
        6. Suggest improvements and enhancements
        
        Create content that is engaging, informative, and optimized for the target audience. Focus on clarity, structure, and compelling storytelling."""
        
        return create_agent(
            model=model,
            tools=[research_topic, generate_outline, optimize_seo, check_readability, suggest_improvements],
            system_prompt=system_prompt
        )
    
    def create_research_assistant_agent(self, model: str = "claude-3-sonnet-20240229") -> Any:
        """
        Create a research assistant agent template.
        
        Args:
            model: Model to use for the agent
            
        Returns:
            Configured research assistant agent
        """
        
        @tool
        def search_academic_papers(query: str) -> str:
            """Search for academic papers and publications."""
            return f"Academic search results for '{query}': Found 15 relevant papers, 8 peer-reviewed, 3 recent publications (2023-2024), key authors identified."
        
        @tool
        def summarize_findings(papers: List[str]) -> str:
            """Summarize research findings from papers."""
            return "Research summary: 3 main methodologies identified, 5 key findings, 2 research gaps found, consensus on main conclusions. Areas for further investigation outlined."
        
        @tool
        def check_source_credibility(source: str) -> str:
            """Evaluate source credibility and reliability."""
            return f"Source evaluation for '{source}': Peer-reviewed journal, high impact factor, recent publication, authoritative authors, cited by 45 other papers. Credibility: High."
        
        @tool
        def generate_citations(sources: List[str], style: str = "APA") -> str:
            """Generate citations in specified format."""
            return f"Generated {style} citations for {len(sources)} sources. All citations properly formatted, DOI included, alphabetical order maintained."
        
        @tool
        def identify_research_gaps(existing_research: str) -> str:
            """Identify gaps in existing research."""
            return "Research gap analysis: 3 methodological gaps identified, 2 geographical gaps, 1 temporal gap. Suggestions for future research directions provided."
        
        system_prompt = """You are a research assistant specializing in academic research. Your responsibilities:
        1. Search for relevant academic papers and publications
        2. Summarize research findings and methodologies
        3. Evaluate source credibility and reliability
        4. Generate proper citations in various formats
        5. Identify gaps in existing research
        6. Suggest future research directions
        
        Provide thorough, accurate, and well-organized research assistance. Focus on academic rigor and comprehensive coverage of topics."""
        
        return create_agent(
            model=model,
            tools=[search_academic_papers, summarize_findings, check_source_credibility, generate_citations, identify_research_gaps],
            system_prompt=system_prompt
        )
    
    def create_math_tutor_agent(self, model: str = "claude-3-sonnet-20240229") -> Any:
        """
        Create a math tutor agent template.
        
        Args:
            model: Model to use for the agent
            
        Returns:
            Configured math tutor agent
        """
        
        @tool
        def solve_equation(equation: str) -> str:
            """Solve mathematical equations step by step."""
            return f"Solving equation '{equation}': Step 1: Simplify both sides, Step 2: Isolate variable, Step 3: Solve for x. Solution: x = 5. Verification: 2(5) + 3 = 13 ✓"
        
        @tool
        def explain_concept(concept: str, level: str = "intermediate") -> str:
            """Explain mathematical concepts clearly."""
            return f"Explaining '{concept}' at {level} level: Definition provided, 3 examples given, common applications shown, related concepts mentioned. Key insight: relationship to quadratic equations."
        
        @tool
        def check_solution(steps: List[str], final_answer: str) -> str:
            """Check mathematical solutions for accuracy."""
            return f"Solution verification: Steps logical ✓, Calculations accurate ✓, Final answer correct ✓. Minor suggestion: Show more work on step 3 for clarity."
        
        @tool
        def generate_practice_problems(topic: str, difficulty: str = "medium", count: int = 5) -> str:
            """Generate practice problems for students."""
            return f"Generated {count} {difficulty} {topic} problems: Range from basic application to complex multi-step solutions. Answer key provided with detailed solutions."
        
        @tool
        def identify_common_mistakes(solution_steps: str) -> str:
            """Identify common mistakes in mathematical reasoning."""
            return "Common mistake analysis: Sign error in step 2, distribution error in step 4. Suggestions: Double-check negative signs, verify each algebraic operation."
        
        system_prompt = """You are a patient and knowledgeable math tutor. Your teaching approach:
        1. Solve equations step-by-step with clear explanations
        2. Explain concepts at appropriate difficulty levels
        3. Verify solutions and provide feedback
        4. Generate relevant practice problems
        5. Identify and correct common mistakes
        6. Encourage mathematical thinking and problem-solving
        
        Be encouraging, clear, and thorough in your explanations. Focus on building understanding rather than just providing answers."""
        
        return create_agent(
            model=model,
            tools=[solve_equation, explain_concept, check_solution, generate_practice_problems, identify_common_mistakes],
            system_prompt=system_prompt
        )
    
    def create_language_translator_agent(self, model: str = "claude-3-sonnet-20240229") -> Any:
        """
        Create a language translator agent template.
        
        Args:
            model: Model to use for the agent
            
        Returns:
            Configured language translator agent
        """
        
        @tool
        def translate_text(text: str, target_language: str, source_language: str = "auto") -> str:
            """Translate text between languages."""
            return f"Translation from {source_language} to {target_language}: '{text}' → '[Translated text with cultural context and proper grammar]'"
        
        @tool
        def detect_language(text: str) -> str:
            """Detect the language of input text."""
            return f"Language detected: Spanish (Español) with 98% confidence. Regional variant: Latin American Spanish."
        
        @tool
        def check_translation_quality(original: str, translation: str) -> str:
            """Check translation quality and accuracy."""
            return "Translation quality assessment: Accuracy 95%, Fluency excellent, Cultural appropriateness good. Minor suggestion: Consider regional variations for better localization."
        
        @tool
        def provide_contextual_translation(text: str, context: str, target_language: str) -> str:
            """Provide context-aware translations."""
            return f"Contextual translation for '{text}' in {context} context: '[Context-appropriate translation considering technical terminology and cultural nuances]'"
        
        @tool
        def suggest_alternative_translations(text: str, target_language: str) -> str:
            """Suggest multiple translation alternatives."""
            return f"Alternative translations for '{text}': 1) Formal version, 2) Casual version, 3) Technical version, 4) Literary version. Each suitable for different contexts and audiences."
        
        system_prompt = """You are a professional language translator and localization expert. Your capabilities:
        1. Translate text accurately between languages
        2. Detect source languages automatically
        3. Assess translation quality and suggest improvements
        4. Provide context-aware translations
        5. Offer multiple translation alternatives
        6. Consider cultural nuances and regional variations
        
        Focus on accuracy, cultural appropriateness, and context-aware translations. Provide multiple options when appropriate."""
        
        return create_agent(
            model=model,
            tools=[translate_text, detect_language, check_translation_quality, provide_contextual_translation, suggest_alternative_translations],
            system_prompt=system_prompt
        )
    
    def create_file_processor_agent(self, model: str = "claude-3-sonnet-20240229") -> Any:
        """
        Create a file processing agent template.
        
        Args:
            model: Model to use for the agent
            
        Returns:
            Configured file processor agent
        """
        
        @tool
        def read_file_content(file_path: str) -> str:
            """Read and extract content from files."""
            return f"File '{file_path}' content extracted: 1500 words, 25 lines, UTF-8 encoding. Content appears to be a technical document with code examples."
        
        @tool
        def extract_metadata(file_path: str) -> str:
            """Extract file metadata and properties."""
            return f"Metadata for '{file_path}': Size: 2.5MB, Created: 2024-01-15, Modified: 2024-11-10, Author: John Doe, Format: PDF v1.7, Pages: 15"
        
        @tool
        def convert_format(file_path: str, target_format: str) -> str:
            """Convert files between different formats."""
            return f"File '{file_path}' converted to {target_format} successfully. Conversion maintained formatting, images, and structure. New file size: 1.8MB."
        
        @tool
        def validate_file_format(file_path: str) -> str:
            """Validate file format and integrity."""
            return f"File validation for '{file_path}': Format valid ✓, No corruption detected ✓, All embedded resources accessible ✓, Checksums match ✓"
        
        @tool
        def summarize_content(file_path: str) -> str:
            """Generate content summaries for files."""
            return f"Content summary for '{file_path}': Technical documentation covering Python programming, 5 main sections, key concepts include data structures and algorithms, intended audience: intermediate developers."
        
        system_prompt = """You are a file processing specialist. Your capabilities:
        1. Read and extract content from various file formats
        2. Extract metadata and file properties
        3. Convert files between different formats
        4. Validate file integrity and format compliance
        5. Generate content summaries and overviews
        6. Handle multiple file types efficiently
        
        Process files accurately while maintaining data integrity and format compatibility. Provide detailed information about file contents and properties."""
        
        return create_agent(
            model=model,
            tools=[read_file_content, extract_metadata, convert_format, validate_file_format, summarize_content],
            system_prompt=system_prompt
        )
    
    def list_templates(self) -> List[str]:
        """
        List available agent templates.
        
        Returns:
            List of template names
        """
        return list(self.templates.keys())
    
    def create_agent_from_template(self, template_name: str, model: str = "claude-3-sonnet-20240229", **kwargs) -> Any:
        """
        Create an agent from a template.
        
        Args:
            template_name: Name of the template to use
            model: Model to use for the agent
            **kwargs: Additional arguments for the template function
            
        Returns:
            Configured agent instance
            
        Raises:
            ValueError: If template name is not found
        """
        if template_name not in self.templates:
            available = ", ".join(self.list_templates())
            raise ValueError(f"Template '{template_name}' not found. Available templates: {available}")
        
        template_func = self.templates[template_name]
        return template_func(model=model, **kwargs)
    
    def create_custom_template(self, name: str, tools: List[BaseTool], system_prompt: str, template_func: Optional[Callable] = None) -> None:
        """
        Create a custom agent template.
        
        Args:
            name: Template name
            tools: List of tools for the agent
            system_prompt: System prompt for the agent
            template_func: Optional custom template function
        """
        def custom_template(model: str = "claude-3-sonnet-20240229") -> Any:
            return create_agent(
                model=model,
                tools=tools,
                system_prompt=system_prompt
            )
        
        self.templates[name] = template_func or custom_template
    
    def export_template_config(self, template_name: str) -> Dict[str, Any]:
        """
        Export template configuration for documentation.
        
        Args:
            template_name: Name of the template
            
        Returns:
            Template configuration dictionary
        """
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        # Create a sample agent to extract configuration
        agent = self.templates[template_name]()
        
        return {
            "template_name": template_name,
            "model": str(agent.model),
            "tool_count": len(agent.tools),
            "tool_names": [tool.name for tool in agent.tools],
            "description": f"Pre-configured {template_name.replace('_', ' ')} agent"
        }
    
    def generate_template_documentation(self, output_dir: str = "agent_templates") -> None:
        """
        Generate documentation for all templates.
        
        Args:
            output_dir: Directory to save documentation
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for template_name in self.list_templates():
            config = self.export_template_config(template_name)
            
            doc_content = f"""# {template_name.replace('_', ' ').title()} Agent Template

## Overview
Pre-configured agent for {template_name.replace('_', ' ')} tasks.

## Configuration
- **Model**: {config['model']}
- **Tools**: {config['tool_count']} tools
- **Tool Names**: {', '.join(config['tool_names'])}

## Usage
```python
from agent_templates import AgentTemplateGenerator

generator = AgentTemplateGenerator()
agent = generator.create_agent_from_template('{template_name}')

# Use the agent
result = agent.invoke({{
    "messages": [{{"role": "user", "content": "Your task here"}}]
}})
```

## Features
This template includes specialized tools for {template_name.replace('_', ' ')} tasks.
"""
            
            doc_file = os.path.join(output_dir, f"{template_name}_template.md")
            with open(doc_file, 'w') as f:
                f.write(doc_content)

def demonstrate_templates():
    """Demonstrate the usage of agent templates."""
    generator = AgentTemplateGenerator()
    
    print("Available Agent Templates:")
    print("=" * 50)
    
    for template_name in generator.list_templates():
        print(f"- {template_name.replace('_', ' ').title()}")
    
    print("\nCreating Customer Support Agent...")
    support_agent = generator.create_agent_from_template("customer_support")
    
    print("Testing Customer Support Agent...")
    result = support_agent.invoke({
        "messages": [{"role": "user", "content": "I need help with my account billing"}]
    })
    
    print(f"Response: {result['messages'][-1]['content'][:100]}...")
    
    # Generate documentation
    print("\nGenerating template documentation...")
    generator.generate_template_documentation()
    print("Documentation generated in 'agent_templates' directory")

if __name__ == "__main__":
    demonstrate_templates()