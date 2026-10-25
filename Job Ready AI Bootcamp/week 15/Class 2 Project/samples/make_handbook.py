"""Build samples/employee_handbook.pdf — a small, fictional 3-page policy PDF for demos and the benchmark.

Run:  python samples/make_handbook.py
"""
from pathlib import Path

from fpdf import FPDF

PAGES = [
    ("Leave Policy",
     "Annual leave: every full-time employee receives 20 days of paid annual leave per calendar year. "
     "Leave accrues monthly at 1.67 days per month. Up to 5 unused days can be carried over to the next year; "
     "anything above 5 days expires on 31 March.\n\n"
     "Sick leave: employees get 12 paid sick days per year. Tell your manager before 10am on the first day. "
     "A doctor's note is required for absences longer than 3 consecutive days.\n\n"
     "Parental leave: the primary caregiver receives 16 weeks of paid leave; the secondary caregiver receives "
     "4 weeks. Parental leave can start up to 2 weeks before the expected birth date."),
    ("IT Security Policy",
     "Passwords must have at least 14 characters and must not be reused across the last 5 passwords. "
     "Multi-factor authentication is mandatory for email, VPN and the HR portal.\n\n"
     "Every company laptop must use full-disk encryption. Lost or stolen devices must be reported to "
     "security@example.com within 24 hours.\n\n"
     "Never paste customer data into external AI tools. Approved internal tools are listed on the intranet."),
    ("Remote Work and Expenses",
     "Employees may work remotely up to 3 days per week with their manager's approval. Core hours are "
     "11:00 to 15:00 local time, when everyone must be reachable.\n\n"
     "A one-time home-office allowance of 300 USD covers a desk, chair or monitor. Receipts must be "
     "submitted within 30 days of purchase.\n\n"
     "Travel: economy class for flights under 6 hours. Meals are reimbursed up to 50 USD per day."),
]


def build(path: Path) -> Path:
    pdf = FPDF()
    for title, body in PAGES:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, f"Acme Corp Employee Handbook - {title}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(0, 6, body)
    pdf.output(str(path))
    return path


if __name__ == "__main__":
    print(build(Path(__file__).with_name("employee_handbook.pdf")))
