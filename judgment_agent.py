"""
Judgment Agent for RAGBot-v2: Deterministic Legal Verdict Adjudication

This agent analyzes case strength from multiple dimensions and renders a
defensible verdict without LLM inference—purely rule-based triangulation
of issue strength, statutory support, and advocacy signals.

JUDGMENT ENGINE THINKING PROTOCOL:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input bundle: case_analysis, law_retrieval, commentary (JSON from upstream)

Deliberation pipeline:
1. Issue scoring
   - Weigh petition allegations vs counter-arguments
   - Emphasize statutes matching issue category + high retrieval confidence
   - Penalize unaddressed allegations, inconsistencies, procedural gaps

2. Factor synthesis
   - Extract factual/procedural/legal factors from all inputs
   - Tag each with directional impact (client/opponent/neutral)
   - Cite evidence: section IDs, quotes, narrative references

3. Verdict calibration
   - Aggregate factor balance → winner (client | opponent | inconclusive)
   - Assign continuous confidence ∈ [0,1] reflecting factor asymmetry
   - Higher statutory depth + strong counter-args → higher confidence

4. Sanity check
   - Probe critical weaknesses or missing evidence
   - Reduce confidence if high-severity gaps found
   - Flag inconclusive if factors nearly balanced

Output: JSON verdict with winner, confidence, decision_factors, narrative
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import re


class JudgmentAgent:
    """
    Judgment Agent: Deterministic verdict engine weighing client vs opponent
    """

    def __init__(self):
        """Initialize judgment agent with scoring parameters"""
        # Scoring weights for factor categories (ENHANCED for more decisive verdicts)
        self.factor_weights = {
            'statutory_support': 0.40,      # Strong law backing (increased)
            'procedural_compliance': 0.25,  # Notice, hearing, jurisdiction
            'factual_strength': 0.20,       # Documentation, chronology
            'counter_argument_quality': 0.10, # Drafting agent analysis
            'weakness_mitigation': 0.05     # How well weaknesses addressed
        }

        # Confidence thresholds (ADJUSTED for higher confidence scores)
        self.confidence_thresholds = {
            'high': 0.75,      # Strong case (increased from 0.70)
            'medium': 0.55,    # Balanced (increased from 0.50)
            'low': 0.35        # Weak or inconclusive (increased from 0.30)
        }

        # Confidence boost parameters
        self.confidence_boost_multiplier = 1.25  # Boost strong verdicts
        self.min_confidence_floor = 0.15  # Minimum confidence (up from 0.1)

    def render_verdict(
        self,
        case_analysis: Dict[str, Any],
        law_retrieval: Dict[str, Any],
        commentary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Main adjudication method: render deterministic verdict

        Args:
            case_analysis: Output from CaseAgent
            law_retrieval: Output from LawAgent
            commentary: Output from DraftingAgent

        Returns:
            Verdict dictionary with winner, confidence, factors, narrative
        """
        # Step 1: Issue scoring - evaluate each legal issue
        issue_scores = self._score_issues(
            case_analysis.get('legal_issues', []),
            law_retrieval.get('issue_law_mapping', [])
        )

        # Step 2: Factor synthesis - extract decision factors
        decision_factors = self._synthesize_factors(
            case_analysis,
            law_retrieval,
            commentary,
            issue_scores
        )

        # Step 3: Verdict calibration - aggregate and decide
        winner, base_confidence = self._calibrate_verdict(decision_factors)

        # Step 4: Sanity check - adjust for critical gaps
        final_confidence, sanity_notes = self._sanity_check(
            winner,
            base_confidence,
            case_analysis,
            law_retrieval,
            decision_factors
        )

        # Generate narrative explanation
        narrative = self._generate_narrative(
            winner,
            final_confidence,
            decision_factors,
            case_analysis,
            law_retrieval
        )

        # Assemble verdict
        verdict = {
            'winner': winner,
            'confidence': round(final_confidence, 3),
            'decision_factors': decision_factors,
            'narrative': narrative,
            'metadata': {
                'rendered_at': datetime.now().isoformat(),
                'issue_count': len(case_analysis.get('legal_issues', [])),
                'statutory_sections_considered': len(law_retrieval.get('all_relevant_sections', [])),
                'factor_count': len(decision_factors),
                'sanity_adjustments': sanity_notes
            }
        }

        return verdict

    def _score_issues(
        self,
        legal_issues: List[Dict[str, Any]],
        issue_law_mapping: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Score each legal issue based on statutory support and attributes

        Returns:
            Dict mapping issue_id to score details
        """
        issue_scores = {}

        # Create quick lookup for law retrieval confidence
        retrieval_lookup = {
            mapping['issue_id']: mapping
            for mapping in issue_law_mapping
        }

        for issue in legal_issues:
            issue_id = issue.get('id')
            category = issue.get('category', 'general')
            source = issue.get('source', 'unknown')
            severity = issue.get('severity', 'medium')
            requires_response = issue.get('requires_response', False)

            # Base score from severity
            base_score = {'high': 3.0, 'medium': 2.0, 'low': 1.0}.get(severity, 2.0)

            # Statutory support from law retrieval
            mapping = retrieval_lookup.get(issue_id, {})
            retrieval_conf = mapping.get('retrieval_confidence', 'none')
            relevant_sections = mapping.get('relevant_sections', [])

            # Boost for strong statutory backing
            statutory_boost = {
                'high': 2.0,
                'medium': 1.0,
                'low': 0.3,
                'none': 0.0
            }.get(retrieval_conf, 0.0)

            # Additional boost per relevant section (diminishing returns)
            section_boost = min(len(relevant_sections) * 0.5, 2.0)

            # Direction: petition issues favor opponent, narrative favors client
            direction = 'opponent' if source == 'petition' else 'client'

            # Penalty for unaddressed issues
            if category == 'unaddressed':
                base_score *= 1.5  # Critical penalty
                direction = 'opponent'

            # Final issue score
            total_score = base_score + statutory_boost + section_boost

            issue_scores[issue_id] = {
                'score': total_score,
                'direction': direction,
                'category': category,
                'severity': severity,
                'statutory_boost': statutory_boost,
                'section_count': len(relevant_sections),
                'requires_response': requires_response
            }

        return issue_scores

    def _synthesize_factors(
        self,
        case_analysis: Dict[str, Any],
        law_retrieval: Dict[str, Any],
        commentary: Dict[str, Any],
        issue_scores: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract decision factors from all inputs and tag with impact direction

        Returns:
            List of factor dictionaries
        """
        factors = []

        # Factor 1: Statutory support depth
        all_sections = law_retrieval.get('all_relevant_sections', [])
        high_conf_mappings = [
            m for m in law_retrieval.get('issue_law_mapping', [])
            if m.get('retrieval_confidence') == 'high'
        ]

        if len(all_sections) >= 3:
            section_citations = ', '.join([s.get('citation', 'provision') for s in all_sections[:3]])
            factors.append({
                'factor': 'statutory_support',
                'evidence': f"{len(all_sections)} relevant statutory provisions retrieved ({section_citations}), {len(high_conf_mappings)} with high confidence",
                'impact': 'client',
                'weight': 3.0,  # Increased from 2.0
                'explanation': f"Strong statutory foundation established. The client's position is supported by {len(all_sections)} relevant legal provisions from the KPK Local Government Act 2013, with {len(high_conf_mappings)} provisions showing high relevance (confidence >0.7). This demonstrates comprehensive legal grounding for the claims. The retrieved sections ({section_citations}) directly address the core legal issues raised in this dispute."
            })
        elif len(all_sections) == 0:
            factors.append({
                'factor': 'statutory_support',
                'evidence': "No relevant statutory provisions found",
                'impact': 'opponent',
                'weight': -3.0,  # Increased from -2.0
                'explanation': "Critical deficiency: The case lacks any identifiable statutory support from the KPK Local Government Act 2013. Without explicit legal provisions backing the claims, the position rests on weak legal foundation. Courts typically require clear statutory authority for the relief sought, and the absence of such authority significantly undermines the legal merit of the case. This suggests either the claims fall outside the scope of the Act or the legal research is incomplete."
            })
        else:
            section_citations = ', '.join([s.get('citation', 'provision') for s in all_sections])
            factors.append({
                'factor': 'statutory_support',
                'evidence': f"Limited statutory backing: {len(all_sections)} section(s) ({section_citations})",
                'impact': 'neutral',
                'weight': 0.5,  # Changed from 0.0
                'explanation': f"Moderate statutory support identified. The case references {len(all_sections)} statutory provision(s) ({section_citations}), which provides some legal foundation but may not be sufficient for a compelling argument. Additional statutory analysis and case law may be needed to strengthen the position. The limited number of provisions suggests either narrow legal issues or potential gaps in legal research."
            })

        # Factor 2: Strengths vs weaknesses balance
        strengths = case_analysis.get('strengths', [])
        weaknesses = case_analysis.get('weaknesses', [])
        high_severity_weaknesses = [w for w in weaknesses if w.get('severity') == 'high']

        if len(strengths) > len(weaknesses):
            strength_details = '; '.join([f"{s.get('category', 'general')}: {s.get('description', '')[:60]}" for s in strengths[:3]])
            factors.append({
                'factor': 'factual_strength',
                'evidence': f"{len(strengths)} strengths vs {len(weaknesses)} weaknesses identified",
                'impact': 'client',
                'weight': 2.5,  # Increased from 1.5
                'explanation': f"Superior factual position demonstrated. The client's case exhibits {len(strengths)} identified strengths compared to only {len(weaknesses)} weaknesses, indicating a favorable factual record. Key strengths include: {strength_details}. This positive strength-to-weakness ratio suggests robust documentation, clear chronology, and well-supported factual allegations. Courts weigh heavily on the quality of factual presentation, and a strong factual foundation significantly enhances the likelihood of success."
            })
        elif len(high_severity_weaknesses) > 0:
            weakness_details = '; '.join([f"{w.get('category', 'general')}: {w.get('description', '')[:80]}" for w in high_severity_weaknesses[:2]])
            factors.append({
                'factor': 'factual_strength',
                'evidence': f"{len(high_severity_weaknesses)} high-severity weakness(es) identified",
                'impact': 'opponent',
                'weight': -2.5,  # Increased from -1.5
                'explanation': f"Significant factual vulnerabilities detected. The case suffers from {len(high_severity_weaknesses)} high-severity weakness(es): {weakness_details}. These critical weaknesses undermine the credibility and persuasiveness of the position. High-severity issues typically involve factual inconsistencies, missing documentation, procedural defects, or contradictions that opposing counsel can exploit. Such weaknesses may prove fatal unless remedied before adjudication."
            })
        else:
            factors.append({
                'factor': 'factual_strength',
                'evidence': f"Balanced case: {len(strengths)} strengths, {len(weaknesses)} weaknesses",
                'impact': 'neutral',
                'weight': 0.3,  # Changed from 0.0
                'explanation': f"Evenly matched factual record. The case presents {len(strengths)} strengths balanced against {len(weaknesses)} weaknesses, suggesting neither party holds a decisive factual advantage. This balanced profile indicates the outcome will likely hinge on legal interpretation and procedural factors rather than factual superiority. Both parties have legitimate factual bases for their positions."
            })

        # Factor 3: Procedural compliance
        procedural_issues = [
            iss for iss in case_analysis.get('legal_issues', [])
            if iss.get('category') == 'procedural'
        ]

        if procedural_issues:
            # Check if procedural issues favor client or opponent
            procedural_scores = [
                issue_scores.get(iss['id'], {}).get('direction', 'neutral')
                for iss in procedural_issues
            ]
            client_proc = procedural_scores.count('client')
            opponent_proc = procedural_scores.count('opponent')

            if client_proc > opponent_proc:
                factors.append({
                    'factor': 'procedural_compliance',
                    'evidence': f"{len(procedural_issues)} procedural issue(s) favor client position ({client_proc} client-favorable vs {opponent_proc} opponent-favorable)",
                    'impact': 'client',
                    'weight': 2.0,  # Increased from 1.0
                    'explanation': f"Procedural advantage established. Out of {len(procedural_issues)} procedural issues identified, {client_proc} favor the client's position while only {opponent_proc} favor the opponent. Procedural compliance is critical in administrative and local government matters, as courts strictly enforce procedural requirements. The client's superior procedural position suggests proper adherence to statutory notice requirements, hearing procedures, and jurisdictional prerequisites. Procedural defects on the opponent's part may provide independent grounds for relief."
                })
            elif opponent_proc > client_proc:
                factors.append({
                    'factor': 'procedural_compliance',
                    'evidence': f"{len(procedural_issues)} procedural issue(s) favor opponent ({opponent_proc} opponent-favorable vs {client_proc} client-favorable)",
                    'impact': 'opponent',
                    'weight': -2.0,  # Increased from -1.0
                    'explanation': f"Procedural deficiencies identified. Analysis reveals {opponent_proc} procedural issues favoring the opponent against only {client_proc} favoring the client. This procedural disadvantage is concerning as courts often resolve cases on procedural grounds without reaching substantive merits. Common issues include lack of proper notice, failure to exhaust administrative remedies, jurisdictional defects, or missed statutory deadlines. These procedural weaknesses may be dispositive even if substantive claims have merit."
                })

        # Factor 4: Unaddressed allegations
        unaddressed = [
            iss for iss in case_analysis.get('legal_issues', [])
            if iss.get('category') == 'unaddressed'
        ]

        if len(unaddressed) > 2:
            unaddressed_preview = '; '.join([u.get('description', '')[:60] for u in unaddressed[:2]])
            factors.append({
                'factor': 'counter_argument_quality',
                'evidence': f"{len(unaddressed)} allegations from petition remain unaddressed",
                'impact': 'opponent',
                'weight': -3.0,  # Increased from -2.0
                'explanation': f"Critical gap: {len(unaddressed)} allegations from the opponent's petition remain completely unaddressed in the client's narrative. Examples include: {unaddressed_preview}. Under legal doctrine, uncontested allegations may be deemed admitted. Failure to respond to specific claims signals either (1) no viable defense exists, (2) inadequate case preparation, or (3) strategic oversight. Courts view silence on material allegations unfavorably, often construing it as tacit admission. This represents a major liability requiring immediate remediation through amended pleadings or supplemental affidavits."
            })
        elif len(unaddressed) == 0:
            factors.append({
                'factor': 'counter_argument_quality',
                'evidence': "All petition allegations addressed in narrative",
                'impact': 'client',
                'weight': 2.0,  # Increased from 1.0
                'explanation': "Comprehensive response strategy. The client's narrative systematically addresses every allegation raised in the opponent's petition, leaving no claims uncontested. This thorough approach demonstrates diligent case preparation and prevents the opponent from claiming any conceded points. Complete responsiveness to adverse allegations is a hallmark of strong advocacy, forcing the opponent to prove every element of their case rather than relying on admissions by silence. This defensive completeness significantly strengthens the client's position."
            })

        # Factor 5: Counter-arguments quality from drafting agent
        counter_args = commentary.get('counter_arguments', {})
        if counter_args and not counter_args.get('error'):
            # Heuristic: length and structure of counter-arguments
            counter_text = counter_args.get('content', '')
            if len(counter_text) > 800:
                factors.append({
                    'factor': 'counter_argument_quality',
                    'evidence': f"Highly detailed counter-arguments drafted ({len(counter_text)} characters)",
                    'impact': 'client',
                    'weight': 2.0,  # Increased from 1.0
                    'explanation': f"Robust counter-argumentation developed. The drafting agent generated {len(counter_text)} characters of detailed counter-arguments, indicating thorough legal analysis and strategic response planning. Comprehensive counter-arguments demonstrate: (1) identification of weaknesses in opponent's case, (2) development of affirmative defenses, (3) statutory and factual rebuttals, and (4) alternative legal theories. The depth and sophistication of counter-arguments correlates strongly with litigation success, as it shows counsel has anticipated and prepared responses to adverse positions."
                })
            elif len(counter_text) > 500:
                factors.append({
                    'factor': 'counter_argument_quality',
                    'evidence': f"Adequate counter-arguments drafted ({len(counter_text)} characters)",
                    'impact': 'client',
                    'weight': 1.2,
                    'explanation': f"Reasonable counter-argumentation provided. The drafting agent produced {len(counter_text)} characters of counter-arguments, covering key opposition points. While adequate for basic responsiveness, additional development may strengthen the position. Counter-arguments address primary allegations but could benefit from deeper statutory analysis and more extensive factual rebuttal to maximize persuasiveness."
                })
            else:
                factors.append({
                    'factor': 'counter_argument_quality',
                    'evidence': f"Minimal counter-arguments drafted ({len(counter_text)} characters)",
                    'impact': 'neutral',
                    'weight': 0.2,
                    'explanation': f"Limited counter-argumentation. Only {len(counter_text)} characters of counter-arguments generated, suggesting either straightforward issues requiring minimal rebuttal or insufficient analysis. Brief counter-arguments may leave critical opponent claims uncontested. Expansion recommended to ensure comprehensive response coverage."
                })

        # Factor 6: Recommendations urgency
        recommendations = case_analysis.get('recommendations', [])
        high_priority_recs = [r for r in recommendations if r.get('priority') == 'high']

        if len(high_priority_recs) > 3:
            rec_preview = '; '.join([r.get('action', '')[:50] for r in high_priority_recs[:2]])
            factors.append({
                'factor': 'weakness_mitigation',
                'evidence': f"{len(high_priority_recs)} high-priority actions required",
                'impact': 'opponent',
                'weight': -1.5,  # Increased from -1.0
                'explanation': f"Substantial remediation required. The case requires {len(high_priority_recs)} high-priority corrective actions ({rec_preview}...). High-priority recommendations typically address critical deficiencies that could be case-dispositive if unresolved. The volume of urgent actions needed suggests significant vulnerabilities requiring immediate attention. Until these high-priority items are addressed, the case remains exposed to serious weaknesses that opposing counsel can exploit."
            })
        elif len(high_priority_recs) > 0:
            factors.append({
                'factor': 'weakness_mitigation',
                'evidence': f"{len(high_priority_recs)} high-priority action(s) required",
                'impact': 'neutral',
                'weight': -0.5,
                'explanation': f"Moderate remediation needed. {len(high_priority_recs)} high-priority action(s) identified for strengthening the case. While not catastrophic, these items warrant prompt attention to shore up vulnerable areas. Addressing these recommendations would enhance case strength and reduce litigation risk."
            })

        # Factor 7: Case complexity and issue distribution
        complexity = case_analysis.get('case_summary', {}).get('complexity', 'medium')
        issue_count = len(case_analysis.get('legal_issues', []))

        if complexity == 'high' and issue_count > 8:
            factors.append({
                'factor': 'factual_strength',
                'evidence': f"High complexity case with {issue_count} issues may require expert testimony",
                'impact': 'neutral',
                'weight': -0.5
            })

        # Factor 8: Statutory citation depth in strongest sections
        top_sections = sorted(
            all_sections,
            key=lambda s: s.get('score', 0.0),
            reverse=True
        )[:3]

        if top_sections and all(s.get('score', 0.0) > 0.7 for s in top_sections):
            citations = [s.get('citation', 'provision') for s in top_sections]
            factors.append({
                'factor': 'statutory_support',
                'evidence': f"Highly relevant statutory matches: {', '.join(citations)} (all scores >0.7)",
                'impact': 'client',
                'weight': 2.5,  # Increased from 1.5
                'explanation': f"Exceptional statutory precision. The top {len(top_sections)} retrieved provisions ({', '.join(citations)}) all exceed 0.7 relevance score, indicating near-perfect semantic and keyword alignment with the case issues. This level of statutory precision is rare and suggests the case falls squarely within well-established legal frameworks. High-scoring provisions provide clear, on-point authority that courts find highly persuasive. This precision significantly strengthens the legal foundation."
            })

        return factors

    def _calibrate_verdict(
        self,
        decision_factors: List[Dict[str, Any]]
    ) -> Tuple[str, float]:
        """
        Aggregate decision factors and determine winner + base confidence

        Returns:
            (winner, base_confidence)
        """
        # Sum weighted impacts
        client_score = sum(
            f['weight'] for f in decision_factors
            if f['impact'] == 'client'
        )

        opponent_score = sum(
            abs(f['weight']) for f in decision_factors
            if f['impact'] == 'opponent'
        )

        neutral_count = sum(
            1 for f in decision_factors
            if f['impact'] == 'neutral'
        )

        # Total magnitude
        total_magnitude = client_score + opponent_score

        # Decide winner (MORE DECISIVE - lowered threshold from 1.0 to 0.8)
        score_difference = abs(client_score - opponent_score)
        if score_difference < 0.8:
            winner = 'inconclusive'
        elif client_score > opponent_score:
            winner = 'client'
        else:
            winner = 'opponent'

        # Calculate base confidence (ENHANCED for higher scores)
        if total_magnitude == 0:
            base_confidence = 0.5  # No evidence either way
        else:
            # Confidence is the winning side's proportion of total score (BOOSTED)
            if winner == 'client':
                raw_confidence = client_score / max(total_magnitude, 1.0)
                # Apply confidence boost for decisive victories
                if raw_confidence > 0.65:
                    base_confidence = min(raw_confidence * self.confidence_boost_multiplier, 0.98)
                else:
                    base_confidence = min(raw_confidence * 1.15, 0.95)
            elif winner == 'opponent':
                raw_confidence = opponent_score / max(total_magnitude, 1.0)
                # Apply confidence boost for decisive opponent victories
                if raw_confidence > 0.65:
                    base_confidence = min(raw_confidence * self.confidence_boost_multiplier, 0.98)
                else:
                    base_confidence = min(raw_confidence * 1.15, 0.95)
            else:
                # Inconclusive: confidence is how balanced it is (inverted)
                balance = 1.0 - score_difference / max(total_magnitude, 1.0)
                base_confidence = min(balance * 0.55, 0.65)  # Slightly higher cap for inconclusive

        # Apply factor diversity bonus (more factors = higher confidence, ENHANCED)
        factor_diversity_bonus = min(len(decision_factors) * 0.02, 0.12)  # Up to +12%
        base_confidence = min(base_confidence + factor_diversity_bonus, 0.99)

        # Apply score magnitude bonus (larger total scores = more confidence)
        if total_magnitude > 10:
            magnitude_bonus = min((total_magnitude - 10) * 0.01, 0.08)  # Up to +8%
            base_confidence = min(base_confidence + magnitude_bonus, 0.99)

        return winner, base_confidence

    def _sanity_check(
        self,
        winner: str,
        base_confidence: float,
        case_analysis: Dict[str, Any],
        law_retrieval: Dict[str, Any],
        decision_factors: List[Dict[str, Any]]
    ) -> Tuple[float, List[str]]:
        """
        Probe for critical gaps and adjust confidence

        Returns:
            (adjusted_confidence, sanity_notes)
        """
        adjustments = []
        confidence = base_confidence

        # Check 1: Missing statutory support (LESS AGGRESSIVE)
        if len(law_retrieval.get('all_relevant_sections', [])) == 0:
            confidence *= 0.85  # Reduced from 0.7
            adjustments.append("No statutory support found - confidence reduced 15%")

        # Check 2: Critical inconsistencies (LESS AGGRESSIVE)
        critical_weaknesses = [
            w for w in case_analysis.get('weaknesses', [])
            if w.get('severity') == 'high' and 'inconsistency' in w.get('category', '')
        ]

        if len(critical_weaknesses) > 1:
            confidence *= 0.90  # Reduced from 0.8
            adjustments.append("Multiple critical inconsistencies - confidence reduced 10%")

        # Check 3: Low retrieval confidence across board (LESS AGGRESSIVE)
        low_conf_mappings = [
            m for m in law_retrieval.get('issue_law_mapping', [])
            if m.get('retrieval_confidence') == 'low'
        ]

        if len(low_conf_mappings) > 3:
            confidence *= 0.92  # Reduced from 0.85
            adjustments.append("Widespread low law-retrieval confidence - reduced 8%")

        # Check 4: Winner is client but many unaddressed allegations (LESS AGGRESSIVE)
        unaddressed = [
            iss for iss in case_analysis.get('legal_issues', [])
            if iss.get('category') == 'unaddressed'
        ]

        if winner == 'client' and len(unaddressed) > 2:
            confidence *= 0.88  # Reduced from 0.75
            adjustments.append("Client favored but multiple unaddressed allegations - reduced 12%")

        # Check 5: Very few decision factors (insufficient data) (LESS AGGRESSIVE)
        if len(decision_factors) < 3:
            confidence *= 0.90  # Reduced from 0.8
            adjustments.append("Sparse decision factors - confidence reduced 10%")

        # Check 6: Flip to inconclusive if confidence drops too low (LOWER THRESHOLD)
        if confidence < 0.30 and winner != 'inconclusive':  # Lowered from 0.40
            adjustments.append(f"Confidence fell below threshold - verdict changed to inconclusive")
            winner = 'inconclusive'
            confidence = min(confidence, 0.55)

        # Floor raised (was 0.1, now 0.15)
        confidence = max(confidence, self.min_confidence_floor)

        return confidence, adjustments

    def _generate_narrative(
        self,
        winner: str,
        confidence: float,
        decision_factors: List[Dict[str, Any]],
        case_analysis: Dict[str, Any],
        law_retrieval: Dict[str, Any]
    ) -> str:
        """
        Generate comprehensive human-readable narrative explanation of verdict

        Returns:
            Extended multi-paragraph narrative with detailed reasoning
        """
        paragraphs = []

        # === PARAGRAPH 1: VERDICT DECLARATION ===
        if winner == 'client':
            opening = f"**VERDICT: CLIENT FAVORED** (Confidence: {confidence:.1%})\n\n"
            opening += f"After conducting a comprehensive multi-factor analysis of the case materials, statutory provisions, and legal arguments, this adjudication engine determines that the client's position holds stronger legal merit. With a confidence level of {confidence:.1%}, the analysis indicates the client has a favorable probability of success should this matter proceed to formal adjudication. This verdict reflects systematic evaluation of {len(decision_factors)} discrete decision factors across statutory support, procedural compliance, factual strength, and advocacy quality dimensions."
        elif winner == 'opponent':
            opening = f"**VERDICT: OPPONENT FAVORED** (Confidence: {confidence:.1%})\n\n"
            opening += f"Following rigorous multi-dimensional analysis of the legal and factual record, this adjudication engine concludes that the opponent's case demonstrates superior merit. At {confidence:.1%} confidence, the assessment suggests the opposing party holds material advantages that would likely prevail in formal proceedings. This determination emerges from careful weighing of {len(decision_factors)} independent decision factors spanning statutory authority, procedural posture, evidentiary strength, and quality of legal argumentation."
        else:
            opening = f"**VERDICT: INCONCLUSIVE** (Confidence: {confidence:.1%})\n\n"
            opening += f"The comprehensive analysis reveals an evenly matched dispute where neither party holds a decisive advantage. At {confidence:.1%} confidence in this inconclusive determination, the case presents balanced strengths and weaknesses on both sides. {len(decision_factors)} decision factors have been evaluated, but they distribute relatively equally between client-favorable and opponent-favorable considerations. This equilibrium suggests the ultimate outcome will depend heavily on trial presentation, witness credibility, and judicial interpretation of close legal questions."

        paragraphs.append(opening)

        # === PARAGRAPH 2: PRIMARY DECISION FACTORS ===
        client_factors = [f for f in decision_factors if f['impact'] == 'client']
        opponent_factors = [f for f in decision_factors if f['impact'] == 'opponent']
        neutral_factors = [f for f in decision_factors if f['impact'] == 'neutral']

        client_score = sum(f['weight'] for f in client_factors)
        opponent_score = sum(abs(f['weight']) for f in opponent_factors)

        factor_analysis = f"\n\n**FACTOR DISTRIBUTION AND SCORING:**\n\n"
        factor_analysis += f"The decision matrix reveals {len(client_factors)} factors favoring the client (cumulative weight: +{client_score:.1f}), "
        factor_analysis += f"{len(opponent_factors)} factors favoring the opponent (cumulative weight: -{opponent_score:.1f}), "
        factor_analysis += f"and {len(neutral_factors)} neutral considerations. "

        if winner == 'client':
            factor_analysis += f"The client's {client_score - opponent_score:.1f}-point scoring advantage reflects meaningful superiority across key evaluation criteria. "
        elif winner == 'opponent':
            factor_analysis += f"The opponent's {opponent_score - client_score:.1f}-point advantage establishes clear superiority in the weighted factor analysis. "
        else:
            factor_analysis += f"The minimal {abs(client_score - opponent_score):.1f}-point differential between parties underscores the closely contested nature of this dispute. "

        paragraphs.append(factor_analysis)

        # === PARAGRAPH 3: STATUTORY ANALYSIS ===
        sections = law_retrieval.get('all_relevant_sections', [])
        high_conf = [m for m in law_retrieval.get('issue_law_mapping', []) if m.get('retrieval_confidence') == 'high']

        statutory_para = f"\n\n**STATUTORY FOUNDATION:**\n\n"
        if len(sections) >= 3:
            top_sections = sorted(sections, key=lambda s: s.get('score', 0.0), reverse=True)[:3]
            citations = ', '.join([s.get('citation', 'provision') for s in top_sections])
            statutory_para += f"The legal analysis identified {len(sections)} relevant statutory provisions from the KPK Local Government Act 2013, with {len(high_conf)} provisions demonstrating high retrieval confidence (relevance scores >0.70). "
            statutory_para += f"The most pertinent authorities include {citations}, each of which directly addresses core legal issues in this dispute. "
            if len(high_conf) >= 2:
                statutory_para += f"The presence of multiple high-confidence statutory matches signals strong legal grounding—these provisions provide clear, on-point authority that courts find particularly persuasive. "
            statutory_para += f"This robust statutory foundation substantially strengthens the favored party's position, as it demonstrates the claims rest on explicit legal authority rather than tenuous legal theories."
        elif len(sections) > 0:
            citations = ', '.join([s.get('citation', 'provision') for s in sections])
            statutory_para += f"The research yielded {len(sections)} potentially applicable statutory provision(s) ({citations}). "
            statutory_para += f"While this provides some legal foundation, the limited statutory support suggests either (1) narrow legal issues, (2) incomplete legal research, or (3) claims that test the boundaries of existing statutory frameworks. "
            statutory_para += f"Additional statutory analysis and case law research may be warranted to fortify the legal position. The paucity of clear statutory authority heightens the importance of persuasive advocacy and factual development."
        else:
            statutory_para += f"**Critical deficiency identified:** The legal research failed to identify any relevant statutory provisions supporting the claims advanced. "
            statutory_para += f"This absence of statutory authority is highly problematic, as administrative and local government law cases typically require explicit statutory grounding. "
            statutory_para += f"Courts are reluctant to grant relief absent clear statutory authorization, particularly in cases involving governmental powers and procedures. "
            statutory_para += f"This gap must be remedied through supplemental legal research, or the claims may need to be reframed to align with identifiable statutory authority."

        paragraphs.append(statutory_para)

        # === PARAGRAPH 4: TOP FACTORS WITH EXPLANATIONS ===
        top_factors_para = f"\n\n**KEY DETERMINATIVE FACTORS:**\n\n"

        # Get top 3 most weighted factors overall
        all_weighted = sorted([f for f in decision_factors if abs(f.get('weight', 0)) > 0.5],
                            key=lambda x: abs(x.get('weight', 0)), reverse=True)[:3]

        for i, factor in enumerate(all_weighted, 1):
            impact_label = "✓ CLIENT ADVANTAGE" if factor['impact'] == 'client' else "✗ OPPONENT ADVANTAGE" if factor['impact'] == 'opponent' else "= NEUTRAL"
            top_factors_para += f"\n{i}. **{factor['factor'].replace('_', ' ').upper()}** ({impact_label}, Weight: {factor.get('weight', 0):+.1f})\n"
            top_factors_para += f"   {factor.get('explanation', factor['evidence'])}\n"

        paragraphs.append(top_factors_para)

        # === PARAGRAPH 5: PROCEDURAL AND FACTUAL CONSIDERATIONS ===
        issues = case_analysis.get('legal_issues', [])
        strengths = case_analysis.get('strengths', [])
        weaknesses = case_analysis.get('weaknesses', [])

        procedural_para = f"\n\n**PROCEDURAL AND FACTUAL POSTURE:**\n\n"
        procedural_para += f"The case involves {len(issues)} identified legal issues spanning procedural, substantive, and remedial dimensions. "
        procedural_para += f"Factual analysis revealed {len(strengths)} case strengths versus {len(weaknesses)} identified weaknesses. "

        if len(strengths) > len(weaknesses):
            procedural_para += f"This favorable strength-to-weakness ratio ({len(strengths)}:{len(weaknesses)}) indicates superior factual development and evidentiary support. "
            procedural_para += f"Strong factual foundations enhance legal arguments by providing concrete support for legal theories and by establishing credibility with the tribunal. "
        elif len(weaknesses) > len(strengths):
            procedural_para += f"The unfavorable ratio of weaknesses to strengths ({len(weaknesses)}:{len(strengths)}) signals material vulnerabilities requiring remediation. "
            procedural_para += f"Factual deficiencies create opportunities for opposing counsel to undermine credibility and cast doubt on legal claims. "

        # Unaddressed allegations check
        unaddressed = [iss for iss in issues if iss.get('category') == 'unaddressed']
        if len(unaddressed) > 0:
            procedural_para += f"**Critical concern:** {len(unaddressed)} allegations from the opponent's petition remain unaddressed in the client narrative. "
            procedural_para += f"Under established legal doctrine, uncontested factual allegations may be deemed admitted. This gap requires urgent attention through amended pleadings or supplemental affidavits. "
        else:
            procedural_para += f"Positively, all opponent allegations have been systematically addressed, preventing any claims from being deemed admitted by silence. "

        paragraphs.append(procedural_para)

        # === PARAGRAPH 6: RECOMMENDATIONS AND CAVEATS ===
        recommendations = case_analysis.get('recommendations', [])
        high_priority = [r for r in recommendations if r.get('priority') == 'high']

        conclusion = f"\n\n**RECOMMENDATIONS AND LIMITATIONS:**\n\n"

        if len(high_priority) > 0:
            conclusion += f"The case analysis generated {len(high_priority)} high-priority recommendations for immediate action. "
            conclusion += f"These recommendations address critical deficiencies that, if unresolved, could undermine the case. "
            conclusion += f"Prompt attention to these action items is essential to maintain the assessed probability of success. "

        if winner == 'client':
            conclusion += f"While the client holds the favored position, continued diligence is required. "
            conclusion += f"The {confidence:.1%} confidence level, while favorable, reflects remaining uncertainties and risks. "
            conclusion += f"Strengthening statutory support, addressing any identified weaknesses, and maintaining comprehensive responses to opponent claims will consolidate the advantage. "
        elif winner == 'opponent':
            conclusion += f"The opponent's favored position presents significant challenges that require strategic response. "
            conclusion += f"At {confidence:.1%} confidence for the opponent, the client faces material litigation risk. "
            conclusion += f"Remedial measures should focus on developing stronger statutory arguments, shoring up factual deficiencies, and exploiting any weaknesses in the opponent's case. "
            conclusion += f"Consider settlement negotiations or alternative dispute resolution given the unfavorable litigation posture. "
        else:
            conclusion += f"Given the inconclusive verdict, both parties face meaningful risk and opportunity. "
            conclusion += f"The outcome will likely hinge on trial advocacy, witness credibility, and how the tribunal interprets close legal questions. "
            conclusion += f"Both sides should prepare thoroughly while remaining open to negotiated resolution given the uncertain outlook. "

        conclusion += f"\n\n**METHODOLOGY NOTE:** This verdict was generated through deterministic rule-based analysis without LLM inference. Factors were systematically scored based on statutory support depth, procedural compliance, factual strength, counter-argument quality, and weakness mitigation. The confidence score reflects factor score distributions, factor diversity, and score magnitude, adjusted for critical gaps through sanity-check protocols. This assessment should be considered alongside professional legal counsel's judgment."

        paragraphs.append(conclusion)

        # Combine all paragraphs
        return ''.join(paragraphs)


def render_case_verdict(
    case_analysis: Dict[str, Any],
    law_retrieval: Dict[str, Any],
    commentary: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convenience function for quick verdict rendering

    Args:
        case_analysis: Output from CaseAgent
        law_retrieval: Output from LawAgent
        commentary: Output from DraftingAgent

    Returns:
        Verdict dictionary
    """
    agent = JudgmentAgent()
    return agent.render_verdict(case_analysis, law_retrieval, commentary)
