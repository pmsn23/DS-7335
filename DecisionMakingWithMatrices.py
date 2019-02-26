# Decision making with Matrices

# This is a pretty simple assingment.  You will do something you do everyday, but today it will be with matrix manipulations.

# The problem is: you and your work firends are trying to decide where to go for lunch. You have to pick a resturant thats best for everyone.  Then you should decided if you should split into two groups so eveyone is happier.

# Displicte the simplictiy of the process you will need to make decisions regarding how to process the data.

# This process was thoughly investigated in the operation research community.  This approah can prove helpful on any number of decsion making problems that are currently not leveraging machine learning.


# You asked your 10 work friends to answer a survey. They gave you back the following dictionary object.
import numpy as np
from scipy.stats import rankdata

def make_dict(names, categories, floatRint):
    # Create People & Restaurant Dictionary with random integer (or) float
    # Sum of Category weight should be 1 for People Matrix
    # floatRInt indicator will instruct the function to execute dirichlet (or)randint function
    
    types = {}
    
    for n in names:
        if floatRint == "F":
            random_prefs = np.random.dirichlet(np.ones(len(categories)),size=1)
            # Need to reshape as dirichlet creates list of list
            random_prefs = random_prefs.reshape(len(categories),)
            random_prefs = np.around(random_prefs, decimals = 4)
        elif floatRint == "I":
            random_prefs = np.random.randint(1, 11, len(categories))
        else:
            print ("Invalid Param .. Kaboom..")
            break
        dict_input = dict(zip(categories, random_prefs))
        types[n] = dict_input
    return types


def convert_list_to_matrix(list):
    # Loop through the dictionary and create matrix
    # p_names/r_names & p_cats/r_cats are the row and column labels
    
    list_of_lists = []
    for names, choices in list.items():
        passing_list = []
        for choice in choices:
            try:
                passing_list.append(choices[choice])
            except KeyError:
                passing_list.append(0)
        list_of_lists.append(passing_list)

    passing_list_array = np.array(list_of_lists)
    return(passing_list_array)
    
def DataProcessing(people, restaurants, names):

    new_people = np.swapaxes(people, 0, 1)
    people_X_restaurants = np.dot(restaurants, new_people)
    restaurants_sum = np.sum(people_X_restaurants, axis=1)
    ranks = rankdata(restaurants_sum).reshape(restaurants_sum.shape)
    rankMatrix = createRankMatrix(names, ranks)
    usr_x_rest_rank = sorted(rankMatrix.items(), key=lambda kv: kv[1])

    return (people_X_restaurants, rankMatrix, usr_x_rest_rank)

def createRankMatrix(names, ranks):
    rankMatrix = {}
    for i in range(len(ranks)):
        name = names[i]
        rank = ranks[i]
        rankMatrix[name] = rank
    return rankMatrix

p_names  = ['Ross', 'Rachel', 'Joey', 'Monica', 'Phoebe','Chandler','Jerry', 'George', 'Kramer', 'Elaine']
p_cats = ['Willingness to travel','Desire for new experience', 'Cost','Vegetarian']

people = make_dict(p_names, p_cats, "F")

M_people = convert_list_to_matrix(people)
print("")
print ("=-=-=-=-=-=-=-=-= People Names =-=-=-=-=-=-=")
print(p_names)
print("")

print("Transform the user data into a matrix(M_people). Keep track of column and row ids")
print ("=-=-=-=-=-=-=-=- People Matrix =-=-=-=-=-=-=")
print (M_people)
print('')

r_names  = ['Flacos', 'PF Changs', 'Madeo', 'Souplantation', 'TGI Friday', 'The Stand','Lamandier','Amelie','Fiesta','Chilis']
r_cats = ['Distance', 'Novelty', 'Cost', 'Vegetarian']

restaurants = make_dict(r_names,r_cats, "I")

M_restaurants = convert_list_to_matrix(restaurants)
print("")
print ("=-=-=-=-=-=-= Restaurants Names =-=-=-=-=-=-=-=-=")
print(r_names)

print("")
print("Transform the restaurant data into a matrix(M_resturants) use the same column index.")
print ("=-=-=-=-=-=-=-=- Resturants Matrix =-=-=-=-=-=-=")
print (M_restaurants)
print('')

print ("The most imporant idea in this project is the idea of a linear combination.")
print("Informally describe what a linear combination is and how it will relate to our resturant matrix.")
print("blah blah blah..")
print("")

M_people_X_restaurants, rankMatrix, M_usr_x_rest_rank = DataProcessing(M_people, M_restaurants, r_names)

print('')
print("Next compute a new matrix (M_usr_x_rest  i.e. an user by restaurant) from all people")
print("=-=-=-=-=-=-= Restaurants X People =-=-=-=-=-=-=-=-=")

print(M_people_X_restaurants)

print("")
print("What does the a_ij matrix represent?")
print("Each Rows represents Restaurants and Column represents the People")
print("")

print ("Sum all columns in M_usr_x_rest to get optimal restaurant for all users.")
print(r_names)
print(np.sum(M_people_X_restaurants, axis=1))
print("")

print("What do the entry’s represent?")
print ("Each entry represents overall score of each restaurants by all users")

print("")
print ("Choose a person and compute(using a linear combination) the top restaurant for them.")
print("What does each entry in the resulting vector represent.")
print ("=-=-=-=-=-= Most Favourite Restaurant of =-=-=-=-=-=")

M_restaurant_max = np.argmax(M_people_X_restaurants, axis=1);
M_people_max = np.argmax(M_people_X_restaurants, axis=0);

for i in range(len(M_people_max)):
    print (p_names[i], "is", r_names[M_people_max[i]])

print("")    
# Which restaurant got max ratings

for i in range(len(M_restaurant_max)):
    print (r_names[i], "got high score from ", p_names[M_restaurant_max[i]])

print("")    
print("Now convert each row in the M_usr_x_rest into a ranking for each user and call it M_usr_x_rest_rank.") 
print("Do the same as above to generate the optimal resturant choice")
print("")
print ("=-=-=-=-=-= Ranks of Restaurants by all People =-=-=-=-=-=-=-=")
print("1: Least & 10: Most Favourite Restaurant")
print("")
print(M_usr_x_rest_rank)
print("")

# Why is there a difference between the two?  What problem arrives?  What does represent in the real world?

# How should you preprocess your data to remove this problem.

# Find user profiles that are problematic, explain why?

print ("Think of two metrics to compute the disatistifaction with the group.")

M_restaurant_min = np.argmin(M_people_X_restaurants, axis=1);
M_people_min = np.argmin(M_people_X_restaurants, axis=0);

print("")
print ("=-=-=-=-=-= Least Favourite Restaurant of =-=-=-=-=-=")
for i in range(len(M_people_min)):
    print (p_names[i], "is", r_names[M_people_min[i]])

print("")
for i in range(len(M_restaurant_min)):
    print (r_names[i], "got low score from ", p_names[M_restaurant_min[i]])


# Should you split in two groups today?

# Ok. Now you just found out the boss is paying for the meal. How should you adjust. Now what is best restaurant?

# Awesome, make the cost weigth to zero and recalculate the rank.

#M_people[:, 2] = 0
#M_people_X_restaurants, rankMatrix, M_usr_x_rest_rank = DataProcessing(M_people, M_restaurants, r_names)
#M_usr_x_rest_rank = sorted(rankMatrix.items(), key=lambda kv: kv[1])
#print ("=-=-=-=-= Restaurants Rank by all People (Boss is paying) =-=-=-=-=-=-=")
#print (M_usr_x_rest_rank)

# Tommorow you visit another team. You have the same restaurants and they told you their optimal ordering for restaurants.  Can you find their weight matrix?
import pdb; pdb.set_trace()