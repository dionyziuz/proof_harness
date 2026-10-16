# The Lonely Runner Conjecture

- **Problem ID**: `lonely-runner`
- **Field**: Diophantine
- **Notability**: Major advance
- **Tags**: Counterexample
- **Prompt Type**: `full_problem`
- **Status**: Open

---

## Problem Statement & Prompt

# The Lonely Runner Conjecture

Consider n runners on a circular track of length 1, all starting at position 0 at time t=0, each running at a distinct positive integer speed. The conjecture states:

    For any such set of n runners, there exists a time t at which every runner is at distance at least 1/(n+1) from the starting position 0.

Here, distance on the circular track means the shorter arc length, i.e. the distance of runner k from 0 at time t is

    ||t * vk|| = min( {t * vk}, 1 - {t * vk} )

where {x} denotes the fractional part of x.

## Your task

Find a counterexample: a list of n >= 2 distinct positive integers v1, ..., vn such that for every time t, at least one runner is at distance strictly less than 1/(n+1) from position 0. In other words, there is no moment at which the starting position is "lonely".

## Submission format

Write your answer to a file containing the speeds as integers separated by spaces or newlines, for example:

    3 7 11 15

Submit your file path using the submit tool.
