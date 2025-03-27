import os

def section_separator(title: str, spacing: int = 2) -> None:
    """Print a section separator.

    :param title: title of the section
    :param spacing: spacing between the sections"""

    # Check whether we are using the terminal
    try:
        separator_length = os.get_terminal_size().columns
    except OSError:
        separator_length = 200

    # Create a centered title and the separator
    separator = "=" * separator_length
    title_padding = (separator_length - len(title)) // 2
    padding = " " * title_padding
    centered_title = f"{padding}{title}{padding}"

    # Print the separator and the centered title
    print("\n" * spacing)  # noqa: T201
    print(f"{separator}\n{centered_title}\n{separator}")
    print("\n" * spacing)