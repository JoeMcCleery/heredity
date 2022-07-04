import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    joint_prob = 1
    # Loop people
    for person in people:
        prob = 1
        # Get number of genes
        num_genes = (1 if person in one_gene else 2 if person in two_genes else 0)
        # Get has trait
        has_trait = (True if person in have_trait else False)
        # Get parents (RIP james and lily)
        father = people[person]["father"]
        mother = people[person]["mother"]
        # If we don't know the state of the person's parents (or they have none)
        if not father and not mother:
            # Get unconditional gene probability
            prob *= PROBS["gene"][num_genes]
        # Calculate inherited gene probability including mutation probability
        else:
            # Get probability of inheriting gene from father
            f_prob = inherit_prob((1 if father in one_gene else 2 if father in two_genes else 0))
            # Get probability of inheriting gene from mother
            m_prob = inherit_prob((1 if mother in one_gene else 2 if mother in two_genes else 0))
            # If inherited 1 gene from parents
            if num_genes == 1:
                # Calc probability of (father giving gene and not mother) or (mother giving gene and not father)
                prob *= f_prob * (1 - m_prob) + m_prob * (1 - f_prob)
            # If inherited 2 genes from parents
            elif num_genes == 2:
                # One gene from father and one gene from mother
                prob *= f_prob * m_prob
            # If inherited no genes
            else:
                # No genes from either parent
                prob *= (1 - f_prob) * (1 - m_prob)
        # Combine probability person has trait given num_genes
        prob *= PROBS['trait'][num_genes][has_trait]

        # Combine with joint probability
        joint_prob *= prob

    return joint_prob


def inherit_prob(num_genes):
    # Return probability of inheriting gene given num_genes in parent
    return 0.5 if num_genes == 1 else 1 - PROBS["mutation"] if num_genes == 2 else PROBS["mutation"]


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    # Loop all people
    for person in probabilities:
        # Get number of genes
        num_genes = (1 if person in one_gene else 2 if person in two_genes else 0)
        # Get has trait
        has_trait = (True if person in have_trait else False)
        # Add gene probability
        probabilities[person]["gene"][num_genes] += p
        # Add trait probability
        probabilities[person]["trait"][has_trait] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    # Loop people
    for person in probabilities:
        # Loop distributions
        for dist in probabilities[person]:
            # Get total of all probabilities in distribution
            total = sum(probabilities[person][dist].values())
            # Normalise each value in distribution
            for value in probabilities[person][dist]:
                probabilities[person][dist][value] *= 1/total


if __name__ == "__main__":
    main()
