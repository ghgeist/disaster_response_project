"""Flask-WTF forms for the Disaster Response application."""

from flask_wtf import FlaskForm
from wtforms import TextAreaField, SubmitField, BooleanField
from wtforms.validators import DataRequired, Length, Regexp


class MessageForm(FlaskForm):
    """
    Form for message classification with comprehensive validation.
    """

    query = TextAreaField(
        "Emergency Message",
        validators=[
            DataRequired(message="Please enter your emergency message"),
            Length(min=3, max=1000, message="Message must be between 3 and 1000 characters"),
            Regexp(r"^[^<>]*$", message="Message cannot contain HTML tags"),
        ],
        render_kw={
            "class": "form-control",
            "placeholder": "Enter an emergency message...",
            "rows": 3,
            "aria-describedby": "query-help",
        },
    )
    use_hierarchy = BooleanField(
        "Use Hierarchy Processing",
        default=False,
        render_kw={
            "class": "form-check-input",
            "aria-describedby": "hierarchy-help",
        },
    )
    submit = SubmitField("Analyze Message", render_kw={"class": "btn btn-primary"})
