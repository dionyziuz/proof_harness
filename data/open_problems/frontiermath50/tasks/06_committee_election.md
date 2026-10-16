# The Core in Approval-Based Committee Elections

- **Problem ID**: `committee-election`
- **Field**: Social choice
- **Notability**: Major advance
- **Tags**: Construction - Finite, Counterexample
- **Prompt Type**: `full_problem`
- **Status**: Solved (2026-09-10)

---

## Problem Statement & Prompt

Your task is to construct an approval-based committee election whose core is empty.

**Setting.** There are m candidates, labelled 1, 2, ..., m, and a target committee size k with 1 <= k < m. There are n voters; each voter i submits an approval set A_i, a subset of the candidates. A committee is a set W of exactly k candidates.

**Core.** Given a committee W, say that a non-empty group of voters S together with a proposal T (any subset of the candidates) blocks W if both of the following hold:

  - |T| / k <= |S| / n   (the group is large enough to "deserve" a committee of size |T|), and
  - |A_i & T| > |A_i & W|  for every voter i in S   (every member of S strictly prefers T to W, i.e. T contains strictly more of their approved candidates than W does).

A committee W is in the core if no group S and proposal T block it. The core of the election is the set of all committees that are in the core.

It is a well-known open problem whether the core is always non-empty. Your task is to find a counterexample: an election instance (a choice of k, m, and a list of approval sets) whose core is empty — that is, every committee of size k is blocked by some group and proposal.

Write your instance to a JSON file and submit its path. The file must contain a single JSON object with the following fields:

  - "k": the committee size (integer, 1 <= k < m).
  - "m": the number of candidates (integer). Candidates are labelled 1..m.
  - "ballots": a non-empty list of objects, one per distinct approval set, each of the form
        {"approve": [<approved candidate labels>], "count": <number of voters with this approval set>}
    where "approve" is a list of distinct integers in 1..m and "count" is a positive integer. The number of voters n is the sum of the counts.

For example, an instance with k = 2, m = 4, and three voters approving {1, 2} and five voters approving {3, 4} would be written as:

  {"k": 2, "m": 4, "ballots": [{"approve": [1, 2], "count": 3}, {"approve": [3, 4], "count": 5}]}

Your submission is correct if and only if the core of the resulting election is empty.
