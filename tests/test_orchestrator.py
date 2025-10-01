"""
Unit tests for orchestrator.py (AgentOrchestrator + WorkflowState)
"""

import pytest
from orchestrator import WorkflowState, WorkflowStatus, AgentOrchestrator


class TestWorkflowState:
    """Test workflow state management"""

    def test_workflow_state_initialization(self):
        """Test workflow state initialization"""
        state = WorkflowState()

        assert state.session_id is not None
        assert state.status == WorkflowStatus.PENDING
        assert state.errors == []
        assert state.warnings == []
        assert state.execution_log == []

    def test_log_step(self):
        """Test logging a workflow step"""
        state = WorkflowState()
        state.log_step('processing', 'success', 'Processing completed')

        assert len(state.execution_log) == 1
        assert state.execution_log[0]['step'] == 'processing'
        assert state.execution_log[0]['status'] == 'success'

    def test_add_error(self):
        """Test adding an error"""
        state = WorkflowState()
        state.add_error('Test error', step='processing')

        assert len(state.errors) == 1
        assert state.errors[0]['message'] == 'Test error'
        assert state.errors[0]['step'] == 'processing'

    def test_add_warning(self):
        """Test adding a warning"""
        state = WorkflowState()
        state.add_warning('Test warning', step='retrieval')

        assert len(state.warnings) == 1
        assert state.warnings[0]['message'] == 'Test warning'

    def test_to_dict(self):
        """Test state serialization"""
        state = WorkflowState()
        state.narrative = "Test narrative"
        state.petition = "Test petition"

        state_dict = state.to_dict()

        assert 'session_id' in state_dict
        assert 'status' in state_dict
        assert 'inputs' in state_dict
        assert state_dict['inputs']['narrative'] == "Test narrative"


class TestAgentOrchestrator:
    """Test agent orchestrator (with mocked dependencies)"""

    def test_orchestrator_initialization(self, mock_vector_store, mock_gemini_client):
        """Test orchestrator initialization"""
        orchestrator = AgentOrchestrator(
            vector_store=mock_vector_store,
            gemini_client=mock_gemini_client
        )

        assert orchestrator.vector_store is not None
        assert orchestrator.gemini_client is not None
        assert orchestrator.document_processor is not None
        assert orchestrator.case_agent is not None
        assert orchestrator.law_agent is not None
        assert orchestrator.drafting_agent is not None

    def test_get_summary(self, mock_vector_store, mock_gemini_client):
        """Test workflow summary generation"""
        orchestrator = AgentOrchestrator(mock_vector_store, mock_gemini_client)
        state = WorkflowState()
        state.status = WorkflowStatus.COMPLETED

        summary = orchestrator.get_summary(state)

        assert 'session_id' in summary
        assert 'status' in summary
        assert 'steps_completed' in summary
        assert 'errors_count' in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
