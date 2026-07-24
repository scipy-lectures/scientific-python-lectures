""" A crude way of getting the solution to the string exercise
"""

def get_cleaned(name_series):

    # Copy the given Series.
    clean_mat_mort_names_solution = name_series.copy()

    # A dictionary of values. The `.keys()` are the original strings, the `.values()`
    # are the replacement strings
    replace_dict = {" ": "_",
                    "(": "",
                    ")": "",
                    ",": "",
                    ".": "",
                    "-": "_"}

    # Loop through the keys and values
    for to_replace, replace in zip(replace_dict.keys(), 
                                   replace_dict.values()):
        # Make the replacements.
        clean_mat_mort_names_solution = (clean_mat_mort_names_solution
                                         .str.replace(to_replace, replace))
    # Remove capitalization
    return clean_mat_mort_names_solution.str.lower()
