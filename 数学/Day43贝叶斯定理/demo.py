p_disease = 0.01
p_positive_give_disease = 0.99
p_positive_give_no_disease = 0.05

p_no_disease = 1-p_disease

p_positive = (
    p_positive_give_disease * p_disease +
    p_positive_give_no_disease * p_no_disease
)

p_disease_given_positive  = (
    p_positive_give_disease * p_disease/p_positive
)

print(f"p(D|+) = {p_disease_given_positive:.4f}")