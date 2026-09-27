from datasets import Dataset

# This code was fully generated with AI
def build_simple_dataset(num_examples=200):
    rows = []
    for i in range(num_examples):
        a = (i * 17) % 180 - 90
        b = (i * 23) % 160 - 80
        c = (i * 29) % 70 + 5
        d = (i * 31) % 50 + 2
        op = i % 6

        if op == 0:
            # Two-step linear expression with multiplication.
            question = f"Compute {a} * {d} + {b}."
            answer = str(a * d + b)
        elif op == 1:
            # Parentheses and subtraction.
            question = f"Compute ({a} + {c}) * ({d} - {b % 9})."
            answer = str((a + c) * (d - (b % 9)))
        elif op == 2:
            # Include squares and mixed signs.
            k = (i % 12) - 6
            question = f"Compute {k}^2 + {a} - {d}."
            answer = str((k * k) + a - d)
        elif op == 3:
            # Exact integer division with a non-trivial numerator.
            q = (i % 11) + 2
            p = ((i * 7) % 40) - 20
            r = (i % 9) - 4
            question = f"Compute ({p} * {q} + {r} * {q} - {b % 5} * {q}) / {q}."
            answer = str(p + r - (b % 5))
        elif op == 4:
            # Three-term arithmetic with negative offsets.
            question = f"Compute ({a} - {b}) + ({c} * {d})."
            answer = str((a - b) + (c * d))
        else:
            # Nested expression with two products.
            x = (i % 13) - 6
            y = ((i * 5) % 13) - 6
            question = f"Compute ({a} + {x}) * ({y} - {d}) - {b}."
            answer = str((a + x) * (y - d) - b)

        rows.append({"question": question, "final_answer": answer})

    return Dataset.from_list(rows)
