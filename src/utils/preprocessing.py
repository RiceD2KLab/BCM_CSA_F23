# 3.0

def subsets(nums):
    """Generete powerset from a input set

    Args:
        nums (list): set of numbers

    Returns:
        list: The power set of the input set.
    """
    
    rez = set()

    def rec(sub):
        if not sub:
            return
        rez.add(tuple(sub))

        for i in sub:
            new_sub = sub.copy()
            new_sub.remove(i)
            rec(new_sub)

    rec(nums)
    rez = list(rez)
    # rez.append([])
    return rez