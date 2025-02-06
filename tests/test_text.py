import pytest
from database.models import Text, Chunk


def test_chunk_content_formatting(db_session, test_teacher):
    """Test preservation of text formatting in chunks"""

    # Create a text first
    text = Text(
        teacher_id=test_teacher.id,
        title="Formatting Test",
        grade_level=5,
        form_name="PROSE",
        type_name="NARRATIVE",
        avg_unit_length="MEDIUM",
    )
    db_session.add(text)
    db_session.commit()

    # Test content with various formatting
    formatted_content = (
        "First paragraph with regular text.\n\n"  # Double line break for paragraphs
        "Second paragraph with special    spacing.\n"  # Multiple spaces
        "\tIndented line.\n"  # Tab character
        "Line with trailing space.  \n"  # Trailing spaces
        "   Line with leading space.\n"  # Leading spaces
        "Line one\n"  # Single line breaks
        "Line two\n"
        "\n"  # Empty lines
        "Final paragraph."
    )

    # Create and save chunk
    chunk = Chunk(
        text_id=text.id, content=formatted_content, word_count=20, is_first=True
    )
    db_session.add(chunk)
    db_session.commit()

    # Retrieve chunk from database
    db_session.refresh(chunk)

    # Verify exact formatting preservation
    assert chunk.content == formatted_content, "Content formatting was not preserved"

    # Verify specific formatting elements
    lines = chunk.content.split("\n")
    assert len(lines) == 10, "Line breaks were not preserved"
    assert lines[0] == "First paragraph with regular text.", "First line mismatch"
    assert lines[1] == "", "Empty line not preserved"
    assert (
        lines[2] == "Second paragraph with special    spacing."
    ), "Multiple spaces not preserved"
    assert lines[3] == "\tIndented line.", "Tab character not preserved"
    assert lines[4] == "Line with trailing space.  ", "Trailing spaces not preserved"
    assert lines[5] == "   Line with leading space.", "Leading spaces not preserved"
