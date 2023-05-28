from tensorflow import constant, reduce_sum

wealth = constant(
    [
        [11, 50],
        [7, 2],
        [4, 60],
        [3, 0],
        [25, 10]
    ]
)

print(wealth.numpy())
print(reduce_sum(wealth, 0))
print(reduce_sum(wealth, 1))
