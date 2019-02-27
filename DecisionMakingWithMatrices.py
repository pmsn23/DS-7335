#import time
#import warnings
import numpy as np
from scipy.stats import rankdata
import matplotlib.pyplot as plt
#from sklearn import cluster
from sklearn.cluster import KMeans #, MiniBatchKMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.decomposition import PCA

import sys
orig_stdout = sys.stdout
f = open('MuthuPalaniHomeWork3.txt', 'w')
sys.stdout = f

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
            random_prefs = np.around(random_prefs, decimals = 2)
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
    # One Stop function to create people x restaurant matrix and the rank..
    new_people = np.swapaxes(people, 0, 1)
    people_X_restaurants = np.dot(restaurants, new_people)
    restaurants_sum = np.sum(people_X_restaurants, axis=1)
    ranks = rankdata(restaurants_sum).reshape(restaurants_sum.shape)
    rankMatrix = createRankMatrix(names, ranks)
    usr_x_rest_rank = sorted(rankMatrix.items(), key=lambda kv: kv[1])

    return (people_X_restaurants, rankMatrix, usr_x_rest_rank)

def createRankMatrix(names, ranks):
    # Create rank from raw score.
    rankMatrix = {}
    for i in range(len(ranks)):
        name = names[i]
        rank = ranks[i]
        rankMatrix[name] = rank
    return rankMatrix

p_names  = ['Ross', 'Rachel', 'Joey', 'Monica', 'Phoebe','Chandler','Jerry', 'George', 'Kramer', 'Elaine']
p_cats = ['Willingness to travel','Desire for new experience', 'Cost', 'Choice of Menu',"Service", 'Environment']

people = make_dict(p_names, p_cats, "F")

M_people = convert_list_to_matrix(people)
print("")
print ("=-=-=-=-=-=-=-=-= People Names =-=-=-=-=-=-=")
print(p_names)
print("")
print("=-=-=-=-=-=-=-=-= Category =-=-=-=-=-=-=")
print(p_cats)
print("")

print("Transform the user data into a matrix(M_people). Keep track of column and row ids")
print ("=-=-=-=-=-=-=-=- People Matrix =-=-=-=-=-=-=")
print (M_people)
print("Each row represents a person and each column is a category in the above order")
print('')

r_names  = ['Flacos', 'PF Changs', 'Madeo', 'Souplantation', 'TGI Friday', 'The Stand','Lamandier','Amelie','Fiesta','Chilis']
r_cats = ['Distance', 'Novelty', 'Cost', "Food-Beverage Options", "Staff", "Ambience"]

restaurants = make_dict(r_names,r_cats, "I")

M_restaurants = convert_list_to_matrix(restaurants)
print("")
print("=-=-=-=-=-=-= Restaurants Names =-=-=-=-=-=-=-=-=")
print(r_names)
print("")
print("=-=-=-=-=-=-=-=-= Category =-=-=-=-=-=-=")
print(r_cats)

print("")
print("Transform the restaurant data into a matrix(M_resturants) use the same column index.")
print("=-=-=-=-=-=-=-=- Restaurants Matrix =-=-=-=-=-=-=")
print(M_restaurants)
print("Each row represents a restaurant and each column is a category in the above order")
print('')

print("The most important idea in this project is the idea of a linear combination.")
print("Informally describe what a linear combination is and how it will relate to our restaurant matrix.")
print("Linear Combination is the process of simplifying two algebric equation so that one variable is eliminated.")
print("In this People vs. Restuarant matrics the weights and ratings are simplified for arriving at the rank ")

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

print("Sum all columns in M_usr_x_rest to get optimal restaurant for all users.")
print(r_names)
print(np.sum(M_people_X_restaurants, axis=1))
print("")

print("What do the entries represent?")
print ("Each entry represents overall score of each restaurants by all users")
print ("This is the raw score out of 100")

print("")
print("Choose a person and compute(using a linear combination) the top restaurant for them.")
print("What does each entry in the resulting vector represent.")
print("=-=-=-=-=-= Most Favorite Restaurant of =-=-=-=-=-=")

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
print("Do the same as above to generate the optimal restaurant choice")
print("")
print("=-=-=-=-=-= Ranks of Restaurants by all People =-=-=-=-=-=-=-=")
print("1: Least & 10: Most Favorite Restaurant")
print("")
print(M_usr_x_rest_rank)
print("")

print("Why is there a difference between the two?")
print("Individual choice and collective choice are closely align if there are no extreme scores (high/low) from individuals")
print("What problem arrives?")
print("High/Low score across on restaurants, for example one person giving high scores to multiple restaurants creates bias ")  
print("What does represent in the real world?")
print("It is difficult to control individual choice as it is based on the individual experience.")
print("This will also open up an opportunity to check if the data is accurate")

print("")
# Code reference:
#https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py

fig, ax = plt.subplots(figsize=(8, 8))
plt.imshow(M_people_X_restaurants)
ax.set_xticks(np.arange(len(r_names)))
ax.set_yticks(np.arange(len(p_names)))

ax.set_xticklabels(r_names)
ax.set_yticklabels(p_names)

plt.setp(ax.get_xticklabels(), rotation=35, ha="right",
         rotation_mode="anchor")

for i in range(len(p_names)):
    for j in range(len(r_names)):
        text = ax.text(j, i, round(M_people_X_restaurants[i, j],2),
                       ha="center", va="center", color="w")

ax.set_title('People Vs. Restaurants Scores') 
fig.tight_layout()
plt.show()
plt.close()

print("How should you preprocess your data to remove this problem.")
print("Could remove the outlier (scored very low) and check if this improves the situation")
print("Identify a common restaurant between the person and the group so that every one can enjoy the meal")
print("Other options discussed are.. ")
print("Overall Score could be used to decide on what restaurants to go!")
print("Identify highest disapproval rating and exclude them from the selection")
print("")
print("Find user profiles that are problematic, explain why?")
print("Heat map created on the matrix could identify the person who made those choices for further action/decision")

# KMeans

def knn2to4(fitMatrix):
    n_clusters = 1 # Initialize, gets incremented inside the loop.
    colors = ['#4EACC5', '#FF9C34', '#4E9A06','#377eb8'] # colors for the plot.

    fig = plt.figure(figsize=(8, 5))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
    plt.suptitle('KMeans with PCA')

    for i in range(3):
        
        ax = fig.add_subplot(1, 3, n_clusters)
        n_clusters = n_clusters+1
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(fitMatrix)
        k_means_cluster_centers = np.sort(kmeans.cluster_centers_, axis=0)
        k_means_labels = pairwise_distances_argmin(fitMatrix, k_means_cluster_centers)

        for k, col in zip(range(n_clusters), colors):
            my_members = k_means_labels == k
            cluster_center = k_means_cluster_centers[k]
            ax.plot(fitMatrix[my_members, 0], fitMatrix[my_members, 1], 'w',
                    markerfacecolor=col, marker='.', markersize=12)
            ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                    markeredgecolor='k', markersize=15)
            
            ax.set_title('KMeans - %i' %n_clusters)
            ax.set_xticks(())
            ax.set_yticks(())

    plt.show()
    plt.close()

    return

# KNN with PCA.
pca = PCA(n_components=2)  
mPeopleXRestaurantsPcaTransform = pca.fit_transform(M_people_X_restaurants)  
knn2to4(mPeopleXRestaurantsPcaTransform)

# Call the function to create KNN with 2 to 4 cluster and visualization.
#knn2to4(M_people_X_restaurants)


# MiniBatchKMeans
'''
batch_size = 45
bk = MiniBatchKMeans(init='k-means++', n_clusters=3, batch_size=batch_size,
                      n_init=10, max_no_improvement=10, verbose=0).fit(M_people_X_restaurants)
mbk_means_cluster_centers = np.sort(mbk.cluster_centers_, axis=0)
mbk_means_labels = pairwise_distances_argmin(M_people_X_restaurants, mbk_means_cluster_centers)

ax = fig.add_subplot(1, 2, 2)
for k, col in zip(range(n_clusters), colors):
    my_members = mbk_means_labels == k
    cluster_center = mbk_means_cluster_centers[k]
    ax.plot(M_people_X_restaurants[my_members, 0], M_people_X_restaurants[my_members, 1], 'w',
            markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=6)
ax.set_title('MiniBatchKMeans')
ax.set_xticks(())
ax.set_yticks(())
'''


print ("Think of two metrics to compute the dissatisfaction with the group.")

M_restaurant_min = np.argmin(M_people_X_restaurants, axis=1);
M_people_min = np.argmin(M_people_X_restaurants, axis=0);

print("")
print("=-=-=-=-=-= Least Favorite Restaurant of =-=-=-=-=-=")
for i in range(len(M_people_min)):
    print (p_names[i], "is", r_names[M_people_min[i]])

print("")
for i in range(len(M_restaurant_min)):
    print (r_names[i], "got low score from ", p_names[M_restaurant_min[i]])


print("Should you split in two groups today?")
print("K-Mean clustering could help decide on the answer, if there is unanimous choice of one individual then Yes.")
print("Otherwise, rely on the recommendation from the clustering results.")

print("")
print("Ok. Now you just found out the boss is paying for the meal. How should you adjust. Now what is best restaurant?")
print("!! Awesome, make the cost weight from people matrix to zero and recalculate the rank.")
print("")

newM_people = M_people
newM_people[:, 2] = 0
M_people_X_restaurants, rankMatrix, M_usr_x_rest_rank = DataProcessing(newM_people, M_restaurants, r_names)
M_usr_x_rest_rank = sorted(rankMatrix.items(), key=lambda kv: kv[1])

print("=-=-=-=-= Restaurants Rank by all People (Boss is paying) =-=-=-=-=-=-=")
print(M_usr_x_rest_rank)

print("")
print("As you can see, the top restaurants choices are same, Cost is not the only deciding factor")
print("")

print("Tommorow you visit another team. You have the same restaurants and they told you their optimal ordering for restaurants.")
print("Can you find their weight matrix?")

sys.stdout = orig_stdout
f.close()

import pdb; pdb.set_trace()
