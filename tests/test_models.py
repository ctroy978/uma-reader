import pytest
from sqlalchemy.exc import IntegrityError
from database.models import (
    User,
    Text,
    Chunk,
    ActiveAssessment,
    Completion,
    CompletionQuestion,
)


class TestUser:
    def test_user_creation(self, db_session):
        """Test creating a new user"""
        user = User(email="new_student@example.com", role_name="STUDENT")
        db_session.add(user)
        db_session.commit()

        assert user.id is not None
        assert user.email == "new_student@example.com"
        assert user.role_name == "STUDENT"
        assert not user.is_deleted

    def test_invalid_email(self, db_session):
        """Test email validation"""
        with pytest.raises(ValueError):
            User(email="invalid-email", role_name="STUDENT")

    def test_invalid_role(self, db_session):
        """Test role validation"""
        with pytest.raises(ValueError):
            User(email="test@example.com", role_name="INVALID_ROLE")

    def test_soft_delete(self, db_session):
        """Test soft delete functionality"""
        user = User(email="to_delete@example.com", role_name="STUDENT")
        db_session.add(user)
        db_session.commit()

        user.soft_delete()
        db_session.commit()

        assert user.is_deleted


class TestText:
    def test_text_creation(self, db_session, test_teacher):
        """Test creating a new text"""
        text = Text(
            teacher_id=test_teacher.id,
            title="Test Text",
            grade_level=5,
            form_name="PROSE",
            type_name="NARRATIVE",
            avg_unit_length="MEDIUM",
        )
        db_session.add(text)
        db_session.commit()

        assert text.id is not None
        assert text.teacher_id == test_teacher.id
        assert text.grade_level == 5
        assert not text.is_deleted

    def test_invalid_grade_level(self, db_session, test_teacher):
        """Test grade level validation"""
        with pytest.raises(ValueError):
            Text(
                teacher_id=test_teacher.id,
                title="Invalid Grade",
                grade_level=13,  # Invalid
                form_name="PROSE",
                type_name="NARRATIVE",
                avg_unit_length="MEDIUM",
            )

    def test_invalid_unit_length(self, db_session, test_teacher):
        """Test unit length validation"""
        with pytest.raises(ValueError):
            Text(
                teacher_id=test_teacher.id,
                title="Invalid Length",
                grade_level=5,
                form_name="PROSE",
                type_name="NARRATIVE",
                avg_unit_length="INVALID",  # Invalid
            )


class TestChunk:
    def test_chunk_creation(self, db_session, test_teacher):
        """Test creating text chunks"""
        # Create a text first
        text = Text(
            teacher_id=test_teacher.id,
            title="Test Text",
            grade_level=5,
            form_name="PROSE",
            type_name="NARRATIVE",
            avg_unit_length="MEDIUM",
        )
        db_session.add(text)
        db_session.commit()  # Commit to ensure text has an ID

        # Create and save first chunk
        chunk1 = Chunk(
            text_id=text.id, content="First chunk", word_count=10, is_first=True
        )
        db_session.add(chunk1)
        db_session.commit()  # Commit to ensure chunk1 has an ID

        # Create and save second chunk
        chunk2 = Chunk(
            text_id=text.id, content="Second chunk", word_count=15, is_first=False
        )
        db_session.add(chunk2)
        db_session.commit()  # Commit to ensure chunk2 has an ID

        # Now link the chunks
        chunk1.next_chunk_id = chunk2.id
        db_session.commit()

        # Refresh from database to ensure we have latest state
        db_session.refresh(chunk1)
        db_session.refresh(chunk2)

        # Verify chain
        assert chunk1.next_chunk_id == chunk2.id
        assert chunk2.next_chunk_id is None
        assert chunk1.is_first
        assert not chunk2.is_first

    def test_invalid_word_count(self, db_session, test_teacher):
        """Test word count validation"""
        text = Text(
            teacher_id=test_teacher.id,
            title="Test Text",
            grade_level=5,
            form_name="PROSE",
            type_name="NARRATIVE",
            avg_unit_length="MEDIUM",
        )
        db_session.add(text)
        db_session.flush()

        with pytest.raises(ValueError):
            Chunk(text_id=text.id, content="Invalid chunk", word_count=0)  # Invalid

    def test_circular_reference_prevention(self, db_session, test_teacher):
        """Test prevention of circular references in chunk chain"""
        text = Text(
            teacher_id=test_teacher.id,
            title="Test Text",
            grade_level=5,
            form_name="PROSE",
            type_name="NARRATIVE",
            avg_unit_length="MEDIUM",
        )
        db_session.add(text)
        db_session.commit()  # Commit to ensure text has an ID

        # Create and save the chunk first
        chunk = Chunk(
            text_id=text.id, content="Test chunk", word_count=10, is_first=True
        )
        db_session.add(chunk)
        db_session.commit()  # Commit to ensure chunk has an ID

        # Try to create circular reference
        chunk.next_chunk_id = chunk.id
        with pytest.raises(ValueError):
            db_session.commit()


class TestActiveAssessment:
    def test_assessment_creation(self, db_session, test_user):
        """Test creating an active assessment"""
        # Create a text first
        text = Text(
            teacher_id=test_user.id,
            title="Test Text",
            grade_level=5,
            form_name="PROSE",
            type_name="NARRATIVE",
            avg_unit_length="MEDIUM",
        )
        db_session.add(text)
        db_session.flush()

        assessment = ActiveAssessment(
            student_id=test_user.id,
            text_id=text.id,
            current_category="literal_basic",
            current_difficulty="basic",
        )
        db_session.add(assessment)
        db_session.commit()

        assert assessment.id is not None
        assert assessment.consecutive_correct == 0
        assert assessment.consecutive_incorrect == 0
        assert assessment.is_active
        assert not assessment.completed

    def test_success_rate_validation(self, db_session, test_user):
        """Test success rate validation"""
        text = Text(
            teacher_id=test_user.id,
            title="Test Text",
            grade_level=5,
            form_name="PROSE",
            type_name="NARRATIVE",
            avg_unit_length="MEDIUM",
        )
        db_session.add(text)
        db_session.flush()

        with pytest.raises(ValueError):
            ActiveAssessment(
                student_id=test_user.id,
                text_id=text.id,
                current_category="literal_basic",
                current_difficulty="basic",
                literal_basic_success=101,  # Invalid
            )

    def test_update_success_rate(self, db_session, test_user):
        """Test updating success rates"""
        text = Text(
            teacher_id=test_user.id,
            title="Test Text",
            grade_level=5,
            form_name="PROSE",
            type_name="NARRATIVE",
            avg_unit_length="MEDIUM",
        )
        db_session.add(text)
        db_session.flush()

        assessment = ActiveAssessment(
            student_id=test_user.id,
            text_id=text.id,
            current_category="literal_basic",
            current_difficulty="basic",
        )
        db_session.add(assessment)
        db_session.flush()

        # Update with correct answer
        assessment.update_success_rate("literal_basic", True)
        assert assessment.literal_basic_success == 100.0
        assert assessment.consecutive_correct == 1
        assert assessment.consecutive_incorrect == 0

        # Update with incorrect answer
        assessment.update_success_rate("literal_basic", False)
        assert assessment.literal_basic_success == 50.0
        assert assessment.consecutive_correct == 0
        assert assessment.consecutive_incorrect == 1
