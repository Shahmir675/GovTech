"""
Drafting Agent for RAGBot-v2: Legal Commentary Generation

This agent generates structured legal commentary including petition critique,
counter-arguments, and recommendations with statutory backing.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime

# Import Gemini client for advanced generation
from gemini_client import EnhancedGeminiClient


class DraftingAgent:
    """
    Drafting Agent: Generates formal legal commentary and analysis
    """

    def __init__(self, gemini_client: EnhancedGeminiClient):
        """
        Initialize Drafting Agent

        Args:
            gemini_client: Initialized EnhancedGeminiClient instance
        """
        self.gemini_client = gemini_client

        # Commentary templates
        self.templates = {
            'petition_critique': """You are a senior legal analyst specializing in local government law in Pakistan.

TASK: Analyze the opponent's petition and provide a structured critique.

OPPONENT'S PETITION SUMMARY:
{petition_summary}

RELEVANT LAW SECTIONS:
{law_sections}

PROVIDE:
1. Strengths of the petition (2-3 points)
2. Weaknesses and gaps (3-5 points with statutory backing)
3. Procedural defects (if any, with section references)
4. Factual inconsistencies (if identified)

Format as structured sections with citations.""",

            'counter_arguments': """You are a legal advocate drafting counter-arguments for a client.

TASK: Draft comprehensive counter-arguments to the opponent's petition.

CASE ISSUES:
{issues}

RELEVANT LAW SECTIONS:
{law_sections}

CLIENT'S STRENGTHS:
{strengths}

OPPONENT'S WEAKNESSES:
{weaknesses}

DRAFT:
1. Counter-argument for each key allegation (with statutory backing)
2. Affirmative defenses (with section references)
3. Procedural objections (if applicable)
4. Alternative interpretations of law (if applicable)

Format as numbered arguments with clear statutory citations.""",

            'recommendations': """You are a legal strategist advising a client on optimal legal strategy.

TASK: Provide strategic recommendations for the client's case.

CASE SUMMARY:
{case_summary}

IDENTIFIED ISSUES:
{issues}

STRENGTHS:
{strengths}

WEAKNESSES:
{weaknesses}

RELEVANT LAW:
{law_sections}

PROVIDE:
1. Immediate actions to strengthen position (3-5 points)
2. Evidence/documentation to gather (specific items)
3. Procedural steps to consider (with statutory basis)
4. Risks and mitigation strategies

Format as prioritized action items with rationale.""",

            'formal_response': """You are a legal draftsman preparing a formal response to a petition.

TASK: Draft a formal legal response document structure.

CASE DETAILS:
{case_summary}

COUNTER-ARGUMENTS:
{counter_arguments}

RELEVANT LAW:
{law_sections}

DRAFT STRUCTURE:
1. Title and parties
2. Preliminary objections (if any)
3. Statement of facts from client's perspective
4. Legal arguments with statutory backing
5. Prayer for relief

Use formal legal language appropriate for KPK courts."""
        }

    def generate_commentary(
        self,
        case_analysis: Dict[str, Any],
        law_retrieval: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Main generation method: Create complete legal commentary

        Args:
            case_analysis: Output from CaseAgent
            law_retrieval: Output from LawAgent

        Returns:
            Structured legal commentary
        """
        # Generate each component
        petition_critique = self._generate_petition_critique(
            case_analysis, law_retrieval
        )

        counter_arguments = self._generate_counter_arguments(
            case_analysis, law_retrieval
        )

        recommendations = self._generate_recommendations(
            case_analysis, law_retrieval
        )

        procedural_guidance = self._generate_procedural_guidance(
            case_analysis, law_retrieval
        )

        # Compile complete commentary
        commentary = {
            'petition_critique': petition_critique,
            'counter_arguments': counter_arguments,
            'recommendations': recommendations,
            'procedural_guidance': procedural_guidance,
            'executive_summary': self._generate_executive_summary(
                case_analysis, law_retrieval, petition_critique, counter_arguments
            ),
            'disclaimer': self._generate_disclaimer(),
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'issue_count': len(case_analysis.get('legal_issues', [])),
                'law_section_count': len(law_retrieval.get('all_relevant_sections', [])),
                'agent_version': 'v2.0'
            }
        }

        return commentary

    def _generate_petition_critique(
        self,
        case_analysis: Dict[str, Any],
        law_retrieval: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate critique of opponent's petition"""
        # Prepare context
        petition_summary = self._format_case_summary(case_analysis)
        law_sections = self._format_law_sections(law_retrieval['all_relevant_sections'][:5])

        # Build prompt
        prompt = self.templates['petition_critique'].format(
            petition_summary=petition_summary,
            law_sections=law_sections
        )

        # Generate using Gemini
        try:
            response = self.gemini_client.model.generate_content(prompt)
            critique_text = response.text

            return {
                'text': critique_text,
                'generated_at': datetime.now().isoformat(),
                'law_sections_used': [s['citation'] for s in law_retrieval['all_relevant_sections'][:5]]
            }

        except Exception as e:
            return {
                'text': f"Error generating critique: {str(e)}",
                'error': True,
                'generated_at': datetime.now().isoformat()
            }

    def _generate_counter_arguments(
        self,
        case_analysis: Dict[str, Any],
        law_retrieval: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate counter-arguments with statutory backing"""
        # Prepare context
        issues = self._format_issues(case_analysis['legal_issues'])
        law_sections = self._format_law_sections(law_retrieval['all_relevant_sections'][:8])
        strengths = self._format_strengths(case_analysis.get('strengths', []))
        weaknesses = self._format_weaknesses(case_analysis.get('weaknesses', []))

        # Build prompt
        prompt = self.templates['counter_arguments'].format(
            issues=issues,
            law_sections=law_sections,
            strengths=strengths,
            weaknesses=weaknesses
        )

        # Generate using Gemini
        try:
            response = self.gemini_client.model.generate_content(prompt)
            arguments_text = response.text

            return {
                'text': arguments_text,
                'generated_at': datetime.now().isoformat(),
                'law_sections_used': [s['citation'] for s in law_retrieval['all_relevant_sections'][:8]]
            }

        except Exception as e:
            return {
                'text': f"Error generating counter-arguments: {str(e)}",
                'error': True,
                'generated_at': datetime.now().isoformat()
            }

    def _generate_recommendations(
        self,
        case_analysis: Dict[str, Any],
        law_retrieval: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate strategic recommendations"""
        # Prepare context
        case_summary = self._format_case_summary(case_analysis)
        issues = self._format_issues(case_analysis['legal_issues'])
        strengths = self._format_strengths(case_analysis.get('strengths', []))
        weaknesses = self._format_weaknesses(case_analysis.get('weaknesses', []))
        law_sections = self._format_law_sections(law_retrieval['all_relevant_sections'][:6])

        # Build prompt
        prompt = self.templates['recommendations'].format(
            case_summary=case_summary,
            issues=issues,
            strengths=strengths,
            weaknesses=weaknesses,
            law_sections=law_sections
        )

        # Generate using Gemini
        try:
            response = self.gemini_client.model.generate_content(prompt)
            recommendations_text = response.text

            # Also include case agent recommendations
            case_recommendations = case_analysis.get('recommendations', [])

            return {
                'strategic_recommendations': recommendations_text,
                'tactical_recommendations': case_recommendations,
                'generated_at': datetime.now().isoformat()
            }

        except Exception as e:
            return {
                'strategic_recommendations': f"Error generating recommendations: {str(e)}",
                'tactical_recommendations': case_analysis.get('recommendations', []),
                'error': True,
                'generated_at': datetime.now().isoformat()
            }

    def _generate_procedural_guidance(
        self,
        case_analysis: Dict[str, Any],
        law_retrieval: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate procedural guidance"""
        procedural_sections = [
            s for s in law_retrieval['all_relevant_sections']
            if 'procedure' in s.get('text', '').lower()
            or 'process' in s.get('text', '').lower()
            or 'rule' in s.get('citation', '').lower()
        ][:3]

        if not procedural_sections:
            # Use general procedural advice
            guidance = {
                'steps': [
                    {
                        'step': 1,
                        'action': 'File formal written response within prescribed time limit',
                        'statutory_basis': 'General procedural rules'
                    },
                    {
                        'step': 2,
                        'action': 'Submit supporting affidavits and documentary evidence',
                        'statutory_basis': 'Rules of evidence'
                    },
                    {
                        'step': 3,
                        'action': 'Request oral hearing if required',
                        'statutory_basis': 'Natural justice principles'
                    }
                ],
                'time_limits': 'Check applicable rules for specific time limits',
                'forum': 'Verify proper forum under KPK Local Government Act 2013'
            }
        else:
            # Extract procedural steps from retrieved sections
            guidance = {
                'steps': [
                    {
                        'step': i + 1,
                        'action': f"Follow procedure under {s['citation']}",
                        'statutory_basis': s['citation']
                    }
                    for i, s in enumerate(procedural_sections)
                ],
                'relevant_sections': [s['citation'] for s in procedural_sections],
                'note': 'Consult complete text of cited sections for detailed procedural requirements'
            }

        return guidance

    def _generate_executive_summary(
        self,
        case_analysis: Dict[str, Any],
        law_retrieval: Dict[str, Any],
        petition_critique: Dict[str, Any],
        counter_arguments: Dict[str, Any]
    ) -> str:
        """Generate executive summary of the complete analysis"""
        case_summary = case_analysis.get('case_summary', {})
        dispute_type = case_summary.get('dispute_type', 'Legal Dispute')
        issue_count = len(case_analysis.get('legal_issues', []))
        complexity = case_summary.get('complexity', 'medium')

        summary_parts = [
            f"**Case Type**: {dispute_type}",
            f"**Complexity**: {complexity.capitalize()}",
            f"**Issues Identified**: {issue_count}",
            f"**Relevant Statutory Provisions**: {len(law_retrieval.get('all_relevant_sections', []))}",
            "",
            "**Key Findings**:",
        ]

        # Add strengths
        strengths = case_analysis.get('strengths', [])
        if strengths:
            summary_parts.append(f"- {len(strengths)} strength(s) identified in client's position")

        # Add weaknesses
        weaknesses = case_analysis.get('weaknesses', [])
        if weaknesses:
            summary_parts.append(f"- {len(weaknesses)} area(s) requiring attention")

        # Add critical issues
        critical_issues = [i for i in case_analysis.get('legal_issues', []) if i.get('severity') == 'high']
        if critical_issues:
            summary_parts.append(f"- {len(critical_issues)} high-priority issue(s) require immediate response")

        summary_parts.extend([
            "",
            "**Primary Recommendations**:",
        ])

        # Add top 3 recommendations
        recommendations = case_analysis.get('recommendations', [])[:3]
        for i, rec in enumerate(recommendations, 1):
            summary_parts.append(f"{i}. {rec.get('action', 'N/A')}")

        return "\n".join(summary_parts)

    def _generate_disclaimer(self) -> str:
        """Generate legal disclaimer"""
        return """**LEGAL DISCLAIMER**

This analysis is AI-generated and provided for informational purposes only. It does not constitute legal advice and should not be relied upon as a substitute for consultation with a qualified legal professional licensed to practice in Pakistan.

Key Limitations:
- This analysis is based on the KPK Local Government Act 2013 and may not cover all applicable laws
- Statutory law is subject to judicial interpretation and amendments
- Factual accuracy depends on information provided by the user
- Procedural rules and time limits must be verified independently
- Jurisdiction and forum must be confirmed with a licensed attorney

**Action Required**: Consult a licensed legal practitioner before taking any legal action based on this analysis.

Generated by: RAGBot-v2 Multi-Agent Legal Assistant
Date: {date}
""".format(date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def _format_case_summary(self, case_analysis: Dict[str, Any]) -> str:
        """Format case summary for prompt"""
        summary = case_analysis.get('case_summary', {})
        lines = [
            f"Dispute Type: {summary.get('dispute_type', 'N/A')}",
            f"Parties: {summary.get('parties', {}).get('user_party', 'User')} vs {summary.get('parties', {}).get('opponent_party', 'Opponent')}",
            f"Issue Count: {summary.get('issue_count', 0)}",
            f"Complexity: {summary.get('complexity', 'medium')}",
        ]
        return "\n".join(lines)

    def _format_issues(self, issues: List[Dict[str, Any]]) -> str:
        """Format issues for prompt"""
        if not issues:
            return "No specific issues identified."

        lines = []
        for i, issue in enumerate(issues[:10], 1):  # Limit to top 10
            lines.append(f"{i}. [{issue.get('category', 'general')}] {issue.get('description', '')[:150]}")

        return "\n".join(lines)

    def _format_law_sections(self, sections: List[Dict[str, Any]]) -> str:
        """Format law sections for prompt"""
        if not sections:
            return "No specific statutory provisions retrieved."

        lines = []
        for i, section in enumerate(sections, 1):
            citation = section.get('citation', 'Unknown')
            text = section.get('sliced_text') or section.get('text', '')
            # Truncate text for prompt efficiency
            text_preview = text[:300] + "..." if len(text) > 300 else text
            lines.append(f"{i}. {citation}\n   {text_preview}\n")

        return "\n".join(lines)

    def _format_strengths(self, strengths: List[Dict[str, str]]) -> str:
        """Format strengths for prompt"""
        if not strengths:
            return "No specific strengths identified."

        lines = []
        for i, strength in enumerate(strengths, 1):
            lines.append(f"{i}. [{strength.get('category', 'general')}] {strength.get('description', '')}")

        return "\n".join(lines)

    def _format_weaknesses(self, weaknesses: List[Dict[str, str]]) -> str:
        """Format weaknesses for prompt"""
        if not weaknesses:
            return "No specific weaknesses identified."

        lines = []
        for i, weakness in enumerate(weaknesses, 1):
            lines.append(f"{i}. [{weakness.get('category', 'general')}] {weakness.get('description', '')}")

        return "\n".join(lines)

    def build_commentary_markdown(self, commentary: Dict[str, Any]) -> str:
        """Return the commentary as a Markdown document."""
        md_lines = [
            "# Legal Commentary Report",
            "",
            f"**Generated**: {commentary['metadata']['generated_at']}",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            commentary['executive_summary'],
            "",
            "---",
            "",
            "## Petition Critique",
            "",
            commentary['petition_critique']['text'],
            "",
            "---",
            "",
            "## Counter-Arguments",
            "",
            commentary['counter_arguments']['text'],
            "",
            "---",
            "",
            "## Strategic Recommendations",
            "",
            commentary['recommendations']['strategic_recommendations'],
            "",
            "### Tactical Actions",
            "",
        ]

        # Add tactical recommendations
        for rec in commentary['recommendations'].get('tactical_recommendations', []):
            md_lines.append(f"- **{rec.get('action', 'N/A')}** (Priority: {rec.get('priority', 'medium')})")
            md_lines.append(f"  - {rec.get('details', '')}")

        md_lines.extend([
            "",
            "---",
            "",
            "## Procedural Guidance",
            "",
        ])

        # Add procedural steps
        guidance = commentary['procedural_guidance']
        for step in guidance.get('steps', []):
            md_lines.append(f"{step['step']}. {step['action']}")
            md_lines.append(f"   - Basis: {step.get('statutory_basis', 'N/A')}")

        md_lines.extend([
            "",
            "---",
            "",
            commentary['disclaimer']
        ])

        return "\n".join(md_lines)

    def export_commentary_markdown(
        self,
        commentary: Dict[str, Any],
        filepath: str
    ) -> str:
        """
        Export commentary to Markdown file and return the document text.

        Args:
            commentary: Generated commentary
            filepath: Output file path
        """
        markdown = self.build_commentary_markdown(commentary)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(markdown)

        return markdown


def generate_commentary(
    case_analysis: Dict[str, Any],
    law_retrieval: Dict[str, Any],
    gemini_client: EnhancedGeminiClient
) -> Dict[str, Any]:
    """
    Convenience function for quick commentary generation

    Args:
        case_analysis: Output from CaseAgent
        law_retrieval: Output from LawAgent
        gemini_client: Initialized Gemini client

    Returns:
        Structured legal commentary
    """
    agent = DraftingAgent(gemini_client)
    return agent.generate_commentary(case_analysis, law_retrieval)
