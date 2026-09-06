"""Query-specific deterministic ground truth for retrieval evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from tests.evals.datasets.retrieval_goldens import rag_retrieval_goldens


EXPECTED_SOURCE = "Comp_Emp_Hand.pdf"


@dataclass(frozen=True)
class ProductionRetrievalGolden:
    """Ground truth for one retrieval query."""

    input: str
    expected_source: str
    expected_pages: frozenset[int]


# Page numbers are 1-based and correspond to the supplied Employee Handbook.
# PyPDFLoader stores page metadata as 0-based integers, so the evaluation
# converts retrieved metadata to this same 1-based representation.
EXPECTED_PAGES_BY_GOLDEN_INDEX: tuple[frozenset[int], ...] = (
    # 1-4 Employment basics
    frozenset({3}), frozenset({3}), frozenset({3}), frozenset({3}),
    # 5-8 Equal opportunity
    frozenset({4}), frozenset({4}), frozenset({4}), frozenset({4}),
    # 9-11 Recruitment
    frozenset({4, 5}), frozenset({5}), frozenset({5}),
    # 12-16 Referrals
    frozenset({5}), frozenset({5}), frozenset({5}), frozenset({5}), frozenset({6}),
    # 17-18 Attendance
    frozenset({6}), frozenset({6}),
    # 19-23 Confidentiality
    frozenset({7}), frozenset({7}), frozenset({7}), frozenset({7}), frozenset({7}),
    # 24-27 Harassment
    frozenset({8}), frozenset({9}), frozenset({9}), frozenset({8}),
    # 28-30 Workplace violence
    frozenset({9}), frozenset({9}), frozenset({9}),
    # 31-32 Workplace safety
    frozenset({10}), frozenset({10, 11}),
    # 33-35 Smoking and drug-free workplace
    frozenset({11}), frozenset({11}), frozenset({11}),
    # 36-39 Internet and digital devices
    frozenset({13}), frozenset({13}), frozenset({14}), frozenset({14}),
    # 40-42 Corporate email
    frozenset({14}), frozenset({14}), frozenset({14}),
    # 43-45 Social media
    frozenset({15}), frozenset({15}), frozenset({15}),
    # 46-47 Conflict of interest
    frozenset({16}), frozenset({16}),
    # 48-50 Employee relationships
    frozenset({16}), frozenset({16}), frozenset({16}),
    # 51-53 Employment of relatives
    frozenset({17}), frozenset({17}), frozenset({17}),
    # 54-55 Workplace visitors
    frozenset({17}), frozenset({17}),
    # 56-57 Solicitation
    frozenset({18}), frozenset({18}),
    # 58-60 Performance management
    frozenset({20}), frozenset({20}), frozenset({21}),
    # 61-62 Training and development
    frozenset({21}), frozenset({21}),
    # 63-66 Work from home
    frozenset({23}), frozenset({23}), frozenset({23}), frozenset({23}),
    # 67-69 Remote working
    frozenset({23}), frozenset({23}), frozenset({23}),
    # 70-72 Employee expenses
    frozenset({24}), frozenset({24}), frozenset({24}),
    # 73-75 Company car
    frozenset({24}), frozenset({25}), frozenset({25}),
    # 76-78 Company-issued equipment
    frozenset({25}), frozenset({26}), frozenset({26}),
    # 79-80 Working hours
    frozenset({26}), frozenset({26}),
    # 81-86 PTO
    frozenset({27}), frozenset({27}), frozenset({27}),
    frozenset({27}), frozenset({27}), frozenset({27}),
    # 87-90 Holidays
    frozenset({27}), frozenset({28}), frozenset({28}), frozenset({28}),
    # 91-94 Sick leave
    frozenset({28}), frozenset({28}), frozenset({28}), frozenset({28}),
    # 95-97 Bereavement
    frozenset({29}), frozenset({29}), frozenset({29}),
    # 98-99 Jury duty and voting
    frozenset({29, 30}), frozenset({30}),
    # 100-103 Parental leave
    frozenset({30}), frozenset({30}), frozenset({30}), frozenset({30}),
    # 104-107 Progressive discipline
    frozenset({31}), frozenset({31}), frozenset({31}), frozenset({32}),
    # 108-111 Resignation
    frozenset({32}), frozenset({32}), frozenset({32}), frozenset({32}),
    # 112-113 Tuition / relocation
    frozenset({32}), frozenset({32}),
    # 114-117 Termination
    frozenset({33}), frozenset({33}), frozenset({33}), frozenset({33}),
    # 118 References
    frozenset({33}),
    # 119-121 Handbook / policy
    frozenset({3}), frozenset({33}), frozenset({34}),
)


if len(EXPECTED_PAGES_BY_GOLDEN_INDEX) != len(rag_retrieval_goldens):
    raise RuntimeError(
        "Retrieval ground truth is out of sync with retrieval_goldens.py: "
        f"expected {len(rag_retrieval_goldens)} page annotations, "
        f"got {len(EXPECTED_PAGES_BY_GOLDEN_INDEX)}."
    )


production_retrieval_goldens: tuple[ProductionRetrievalGolden, ...] = tuple(
    ProductionRetrievalGolden(
        input=golden.input,
        expected_source=EXPECTED_SOURCE,
        expected_pages=pages,
    )
    for golden, pages in zip(
        rag_retrieval_goldens,
        EXPECTED_PAGES_BY_GOLDEN_INDEX,
        strict=True,
    )
)
