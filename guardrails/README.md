# Part D Guardrails and Determinism

## What I tested
I ran the same prompts multiple times with temperature set to 0 and top p set to 1.
I used a fixed seed and also a per prompt seed.
I compared outputs by hashing a normalized form of the response.

## How I verified determinism
For each prompt I ran repeated requests and checked that all output hashes matched.
If any hash differed then the prompt was flagged as nondeterministic.

## Validation guardrails
I added lightweight validation for expected output formats.
The script supports built in validators for A B C D only, yes or no, numbers, and JSON with required keys.
A run is considered successful only if the request succeeds and the output passes validation.

## Where nondeterminism can still persist
Some backends may ignore or not support a seed setting.
Token streaming can include variable whitespace and newlines across runs.
Hardware and low level kernels can introduce small timing variation that may impact generation in some engines.
Server side configuration such as threading and caching can affect outputs.

## How to run
Example for A B C D validation
python guardrails/validate.py --validator single_letter_abcd --repeats 5

## Example for yes or no validation
python guardrails/validate.py --validator yes_no --repeats 5

## Example using per prompt seed
python guardrails/validate.py --seed_mode per_prompt --repeats 5

## Outputs
The script writes guardrails/determinism_report.csv with one row per run including hashes and basic timing.
