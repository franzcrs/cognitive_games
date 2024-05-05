import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import matplotlib.pyplot as plt

def compute_new_node_attribute(node_data, centroid_data, similarity_type='cosine'):
    # Compute the cosine similarity for each vector in cluster_data
    new_node_attribute = np.dot(node_data, centroid_data) / (np.linalg.norm(node_data) * 
                                                         np.linalg.norm(centroid_data)) if similarity_type == 'cosine' else np.linalg.norm(node_data - centroid_data)

    return new_node_attribute

def compute_nodes_attribute(cluster_file, centroid_type='mean', similarity_type='cosine'):
    # Load the cluster data from the text file
    cluster_data = np.loadtxt(cluster_file)

    # Extract the nodes
    # nodes = cluster_data[:-1]  # Exclude the last row which represents the centroid

    # Compute centroid from the cluster data by taking the mean or median
    centroid_data = np.mean(cluster_data, axis=0) if centroid_type == 'mean' else np.median(cluster_data, axis=0)

    # Compute the cosine similarity for each vector in cluster_data
    nodes_attribute = [np.dot(cluster_data[i], centroid_data) / (np.linalg.norm(cluster_data[i]) * np.linalg.norm(centroid_data)) 
                       for i in range(len(cluster_data))] if similarity_type == 'cosine' else [np.linalg.norm(cluster_data[i] - centroid_data) 
                                                                                               for i in range(len(cluster_data))]

    # Convert the list to a numpy array
    nodes_attribute = np.array(nodes_attribute)

    return nodes_attribute, centroid_data

def visualize_meas_central_tend(nodes_attributes, new_node_attributes, similarity_type='cosine'):
    nodes_attributes_1 = nodes_attributes[0]
    nodes_attributes_2 = nodes_attributes[1]
    nodes_attributes_3 = nodes_attributes[2]
    new_node_attribute_1 = new_node_attributes[0]
    new_node_attribute_2 = new_node_attributes[1]
    new_node_attribute_3 = new_node_attributes[2]
    
    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])  # Set the aspect ratio of the plot
    ax.view_init(elev=20, azim=-45)  # Set the viewing angle of the plot
    ax.dist = 12  # Set the distance between the plot and the camera

    # # Compute the mean and standard deviation of the nodes_attributes_1 array
    # mean = np.mean(nodes_attributes_1)
    # std = np.std(nodes_attributes_1)
    # # Generate the range of values for x axis in the 95% confidence interval
    # x = np.linspace(mean - 2*std, mean + 2*std, 100)
    # # Compute the probability density function using the mean and standard deviation
    # y = (1/(std * np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mean)/std)**2)
    # # Normalize the probability density function
    # y = y / (np.max(y)*2)
    # # Plot the normal distribution curve in the x and y axis of the 3D plot
    # ax.plot(x, y, np.zeros(len(x)), color='blue')#, label='Normal Distribution')

    # Compute the histogram of the nodes_attributes_1 array
    hist, bins = np.histogram(nodes_attributes_1, bins=50, density=True)
    bins = bins + (bins[1] - bins[0])/2  # Shift the bins to the center
    bins = bins[:-1] # Remove the last bin
    # Superior limit for the histogram
    limit_sup = 1 if similarity_type == 'cosine' else np.max(nodes_attributes_2)
    # Plot the histogram in the x and y axis of the 3D plot
    ax.plot(bins, hist*limit_sup / np.max(hist), np.zeros(len(bins)), c='royalblue', linewidth=0.5, alpha=0.7)
    # Draw the median of the histogram with a dotted line
    median_1 = np.median(nodes_attributes_1)
    # ax.plot([median_1, median_1], [0, 1], [0, 0], linestyle='--', color='cornflowerblue', linewidth=2.5, label='Median Cluster 1', alpha=0.7)
    # Draw the mode of the histogram with a dotted line
    mode_1 = bins[np.argmax(hist)]
    ax.plot([mode_1, mode_1], [0, limit_sup], [0, 0], linestyle='--', color='cornflowerblue', linewidth=2.5, label='Mode Cluster 1', alpha=0.7)
    # Normalize the histogram values
    hist = hist / np.sum(hist)
    # Find the index of the 5% percentile of the histogram
    index = np.where(np.cumsum(hist) > 0.05)[0][0]
    # Draw the 5% percentile of the histogram with a dotted line
    percentile5_1 = bins[index]
    ax.plot([percentile5_1, percentile5_1], [0, limit_sup], [0, 0], linestyle='--', color='slategrey', linewidth=2.5, label='5% Percentile Cluster 1', alpha=0.7)

    # Plot the nodes from the first set
    ax.scatter(nodes_attributes_1, np.zeros(len(nodes_attributes_1)), np.zeros(len(nodes_attributes_1)), c='royalblue', label='Cluster 1')
    # Plot the new node in the axis of cluster 1
    ax.scatter(new_node_attribute_1, 0, 0, c='red', marker='x', s=100, label='New Node Similarity with Cluster 1')
    
    
    # # Compute the mean and standard deviation of the nodes_attributes_2 array
    # mean = np.mean(nodes_attributes_2)
    # std = np.std(nodes_attributes_2)
    # # Generate the range of values for x axis in the 95% confidence interval
    # y = np.linspace(mean - 2*std, mean + 2*std, 100)
    # # Compute the probability density function using the mean and standard deviation
    # z = (1/(std * np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mean)/std)**2)
    # # Normalize the probability density function
    # z = z / (np.max(z)*2)
    # # Plot the normal distribution curve in the x and y axis of the 3D plot
    # ax.plot(np.zeros(len(y)), y, z, color='red')#, label='Normal Distribution')

    # Compute the histogram of the nodes_attributes_1 array
    hist, bins = np.histogram(nodes_attributes_2, bins=50, density=True)
    bins = bins + (bins[1] - bins[0])/2  # Shift the bins to the center
    bins = bins[:-1] # Remove the last bin
    # Superior limit for the histogram
    limit_sup = 1 if similarity_type == 'cosine' else np.max(nodes_attributes_3)
    # Plot the histogram cure in the y and z axis of the 3D plot
    ax.plot(np.zeros(len(bins)), bins, hist*limit_sup / np.max(hist), c='firebrick', linewidth=0.5, alpha=0.7)
    # Draw the median of the histogram with a dotted line
    median_2 = np.median(nodes_attributes_2)
    # ax.plot([0, 0], [median_2, median_2], [0, 1], linestyle='--', color='indianred', linewidth=2.5, label='Median Cluster 2', alpha=0.7)
    # Draw the mode of the histogram with a dotted line
    mode_2 = bins[np.argmax(hist)]
    ax.plot([0, 0], [mode_2, mode_2], [0, limit_sup], linestyle='--', color='indianred', linewidth=2.5, label='Mode Cluster 2', alpha=0.7)
    # Normalize the histogram values
    hist = hist / np.sum(hist)
    # Find the index of the 5% percentile of the histogram
    index = np.where(np.cumsum(hist) > 0.05)[0][0]
    # Draw the 5% percentile of the histogram with a dotted line
    percentile5_2 = bins[index]
    ax.plot([0, 0], [percentile5_2, percentile5_2], [0, limit_sup], linestyle='--', color='slategrey', linewidth=2.5, label='5% Percentile Cluster 2', alpha=0.7)

    # Plot the nodes from the second set
    ax.scatter(np.zeros(len(nodes_attributes_2)), nodes_attributes_2, np.zeros(len(nodes_attributes_2)), c='firebrick', label='Cluster 2')
    # Plot the new node in the axis of cluster 2
    ax.scatter(0, new_node_attribute_2, 0, c='red', marker='x', s=100, label='New Node Similarity with Cluster 2')

    # # Compute the mean and standard deviation of the nodes_attributes_2 array
    # mean = np.mean(nodes_attributes_3)
    # std = np.std(nodes_attributes_3)
    # # Generate the range of values for x axis in the 95% confidence interval
    # z = np.linspace(mean - 2*std, mean + 2*std, 100)
    # # Compute the probability density function using the mean and standard deviation
    # x = (1/(std * np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mean)/std)**2)
    # # Normalize the probability density function
    # x = x / (np.max(x)*2)
    # # Plot the normal distribution curve in the x and y axis of the 3D plot
    # ax.plot(x, np.zeros(len(z)), z, color='green')#, label='Normal Distribution')

    # Compute the histogram of the nodes_attributes_1 array
    hist, bins = np.histogram(nodes_attributes_3, bins=50, density=True)
    bins = bins + (bins[1] - bins[0])/2  # Shift the bins to the center
    bins = bins[:-1] # Remove the last bin
    # Superior limit for the histogram
    limit_sup = 1 if similarity_type == 'cosine' else np.max(nodes_attributes_1)
    # Plot the histogram curve in the z and x axis of the 3D plot
    ax.plot(hist*limit_sup / np.max(hist), np.zeros(len(bins)), bins, c='darkolivegreen', linewidth=0.5, alpha=0.7)
    # Draw the median of the histogram with a dotted line
    median_3 = np.median(nodes_attributes_3)
    # ax.plot([0, 1], [0, 0], [median_3, median_3], linestyle='--', color='olivedrab', linewidth=2.5, label='Median Cluster 3', alpha=0.7)
    # Draw the mode of the histogram with a dotted line
    mode_3 = bins[np.argmax(hist)]
    ax.plot([0, limit_sup], [0, 0], [mode_3, mode_3], linestyle='--', color='olivedrab', linewidth=2.5, label='Mode Cluster 3', alpha=0.7)
    # Normalize the histogram values
    hist = hist / np.sum(hist)
    # Find the index of the 5% percentile of the histogram
    index = np.where(np.cumsum(hist) > 0.05)[0][0]
    # Draw the 5% percentile of the histogram with a dotted line
    percentile5_3 = bins[index]
    ax.plot([0, limit_sup], [0, 0], [percentile5_3, percentile5_3], linestyle='--', color='slategrey', linewidth=2.5, label='5% Percentile Cluster 3', alpha=0.7)

    # Plot the nodes from the third set
    ax.scatter(np.zeros(len(nodes_attributes_3)), np.zeros(len(nodes_attributes_3)), nodes_attributes_3, c='darkolivegreen', label='Cluster 3')
    # Plot the new node in the axis of cluster 3
    ax.scatter(0, 0, new_node_attribute_3, c='red', marker='x', s=100, label='New Node Similarity with Cluster 3')

    # Set labels and legend
    ax.set_xlabel('Distance with cluster 1 centroid')
    ax.set_ylabel('Distance with cluster 2 centroid')
    ax.set_zlabel('Distance with cluster 3 centroid')
    # ax.legend()
    # Show the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

    percentile5 = [percentile5_1, percentile5_2, percentile5_3]
    mode = [mode_1, mode_2, mode_3]
    median = [median_1, median_2, median_3]
    return percentile5, mode, median
    
def find_similar_class(new_feature_vector, file_paths, class_labels):
    similarity_type = 'cosine'
    nodes_attributes = []
    new_node_attributes = []
    for file_path in file_paths:
        nodes_attribute, centroid_vector = compute_nodes_attribute(file_path, centroid_type='mean', similarity_type=similarity_type)
        new_node_attribute = compute_new_node_attribute(new_feature_vector, centroid_vector, similarity_type=similarity_type)
        nodes_attributes.append(nodes_attribute)
        new_node_attributes.append(new_node_attribute)
    percentile5, mode, median = visualize_meas_central_tend(nodes_attributes, new_node_attributes, similarity_type=similarity_type)
    similarity_list = [new_node_attribute if new_node_attribute > percentile5[index] else 0 for index, new_node_attribute in enumerate(new_node_attributes)]
    class_index = similarity_list.index(max(similarity_list))
    return class_index, class_labels[class_index]

# Example usage
file_paths = [
    'L_vectors.txt',
    'C_vectors.txt',
    'None_vectors.txt'
]

for file_path in file_paths:
    with open(file_path, 'r') as file:
        data_string = file.read()
        data_array = np.fromstring(data_string, sep=' ')
        print(data_array)

new_vector_data = "-0.228461 -0.328493 -0.373872 -0.306060 0.302574 -0.084282 0.047661 -0.180656 -0.270126 0.643528 -0.292117 0.249690 -0.222214 0.440414 -0.076914 -0.122683 -0.000000 -0.341081 -0.243594 -0.053192 1.121171 0.248480 0.094270 -0.035677 -0.000000 1.405232 -0.000000 -0.270873 -0.354515 -0.133253 0.297672 -0.373697 1.212552 0.055255 -0.000000 0.276513 -0.127636 -0.374951 0.439935 -0.069769 -0.136323 1.642792 -0.359344 -0.202909 0.407588 -0.313653 0.109877 2.507822 1.532556 -0.364985 0.764646 0.589244 -0.221335 -0.374868 0.133053 0.175679 -0.346454 -0.000000 -0.369767 -0.000000 -0.319788 0.362980 0.565868 -0.019167 -0.107778 0.837651 -0.355045 0.161848 -0.195014 -0.355345 -0.369550 0.122935 -0.080841 0.893097 -0.199746 -0.372847 2.188787 -0.124203 -0.242723 -0.000000 1.108456 0.089289 -0.170956 -0.210547 -0.098163 0.120632 -0.372623 -0.284373 -0.000000 -0.265928 -0.000000 -0.350241 -0.320470 -0.000000 -0.000000 -0.165828 1.419124 -0.029503 -0.000000 -0.303800 0.241498 0.073272 -0.367291 -0.369826 -0.365896 -0.270451 -0.352813 -0.374999 0.407609 0.724162 0.155795 -0.095490 -0.044327 0.023158 -0.329749 -0.321868 1.038376 0.306398 -0.374991 -0.366081 -0.000000 -0.017307 -0.209255 -0.338102 0.248721 0.403249 0.082157 -0.090489 0.455076 -0.322607 -0.000000 -0.000000 1.585254 0.398259 1.241123 -0.368188 -0.187211 0.036180 0.000162 -0.342403 -0.364502 -0.241158 -0.102269 0.446548 -0.111911 -0.278764 -0.365983 0.446947 0.578691 -0.374109 0.468520 -0.368744 -0.329398 -0.229269 -0.173369 0.818426 -0.285302 0.403379 0.949567 0.421919 -0.000000 -0.320207 -0.223660 0.575271 -0.308825 0.198744 -0.260366 0.132472 0.856529 0.036277 -0.000000 -0.243812 -0.087305 -0.057796 -0.374469 -0.000000 -0.132406 -0.357191 -0.046861 -0.374340 0.105466 -0.239628 -0.092775 0.489677 -0.031150 -0.000000 -0.355460 -0.184415 0.716426 -0.273561 0.678282 1.407931 0.513435 -0.165167 0.022683 -0.119998 -0.000000 -0.373582 0.358363 -0.349246 -0.107344 -0.354100 0.157483 -0.020728 0.049616 -0.000000 -0.370058 -0.000000 0.232912 0.408423 -0.299868 -0.071469 -0.254429 0.590546 -0.369431 -0.202366 1.656623 -0.196518 -0.276257 -0.065160 -0.124647 -0.365474 -0.358221 -0.374218 1.102416 -0.209844 -0.258619 -0.369269 -0.060800 -0.107102 1.456553 1.731008 -0.158231 0.981715 -0.198532 -0.284037 -0.210022 0.613223 -0.370946 0.073217 0.041208 -0.294630 -0.000000 -0.302051 0.004590 0.533827 -0.340297 -0.054013 0.326391 -0.002861 0.003245 0.045280 -0.374942 -0.346221 -0.000000 0.617055 -0.309367 -0.146911 -0.357066 -0.114250 -0.312460 0.555516 0.335223 -0.258624 -0.228512 -0.349200 0.604809 0.191714 0.094547 0.633279 -0.246090 -0.304640 -0.340600 1.075864 0.899401 -0.164856 0.193060 -0.085001 -0.230484 -0.218712 -0.301161 -0.344212 -0.296343 -0.346719 -0.256590 -0.099635 -0.240609 -0.259399 1.046234 0.628625 0.987513 -0.000000 -0.338788 -0.291719 0.743802 -0.000000 -0.232946 0.459698 -0.000000 0.533237 0.461080 0.464819 -0.000000 1.239580 -0.005761 0.053140 -0.351312 -0.360320 -0.279014 -0.224196 0.640028 0.988545 -0.373505 -0.301872 1.682959 0.917870 -0.019702 0.151431 -0.219594 0.101189 1.564982 -0.373465 0.990527 -0.363468 0.353737 2.389206 -0.348714 0.259729 -0.041751 -0.149045 -0.255393 0.449676 0.508684 0.008859 1.645771 -0.285483 -0.233603 -0.288789 1.135978 0.227190 -0.344978 -0.361884 -0.363848 -0.035591 -0.054743 0.077475 0.159291 -0.259406 -0.317700 0.230803 0.080496 0.607874 0.272326 1.280633 0.202334 1.333784 -0.321721 -0.371122 1.461621 -0.304420 0.062409 -0.000000 -0.177605 -0.000000 -0.115875 0.181323 -0.032059 -0.227621 0.662605 -0.325017 -0.132547 -0.139058 0.465669 -0.323012 0.838431 0.514345 0.852585 -0.178398 -0.126768 -0.373624 -0.374411 -0.226515 1.016109 -0.069461 0.560575 0.232005 -0.203798 -0.031098 0.096749 -0.271887 -0.313002 0.653507 0.460951 0.389260 -0.336601 -0.240815 2.512089 -0.215795 -0.176513 -0.143077 -0.259214 -0.349403 0.589911 -0.335608 0.350988 0.269524 0.263816 -0.224700 -0.343244 0.011452 -0.371344 -0.000000 -0.162773 -0.369978 1.248605 0.158571 -0.145728 -0.111897 -0.014422 1.197368 0.737875 1.344440 1.612956 0.155525 -0.354665 -0.342115 -0.321578 -0.136496 -0.290336 -0.079757 -0.035157 -0.000000 -0.347894 -0.335498 -0.328010 -0.244796 -0.000000 -0.000000 -0.064621 0.647600 -0.015000 -0.366541 0.258973 0.910027 0.214698 -0.374762 -0.125179 -0.369987 0.105619 -0.374929 0.675006 -0.238749 0.076147 -0.372965 -0.366179 0.226504 0.157479 -0.322913 -0.171114 -0.203756 -0.370584 -0.352128 -0.343154 -0.357214 0.112497 0.921799 1.414851 0.262186 -0.328708 -0.230437 0.111668 -0.301701 -0.000000 0.051502 -0.218517 0.523510 -0.000000 -0.175469 -0.374777 -0.064070 -0.362678 -0.169641 -0.102579 -0.000000 -0.202982 0.629110 0.608381 -0.234050 0.760472 1.538510 -0.322008 -0.345362 -0.320135 -0.333145 -0.313968 -0.368629 0.091461 -0.035605 -0.317668 0.431428 0.132583 -0.025611 0.111347 -0.000000 -0.167435 -0.261712 -0.296952 -0.130929 0.128251 -0.374815 -0.021390 -0.348189 0.170440 0.300992 -0.333193 -0.033248 0.191224 0.184091 1.659748 0.569827 0.294739 -0.302935 -0.374092 -0.366854 -0.357911 -0.038123 -0.034224 -0.373936 -0.020396 0.016691 -0.234039 -0.113007 -0.089275 -0.000000 -0.372852 -0.000000 0.264664 0.828552 -0.358713 -0.208623 0.177775 -0.071268 0.301080 0.253459 1.965421 0.173157 -0.297693 -0.082249 0.004471 -0.026019 -0.107226 0.308593 0.248611 0.806785 -0.000000 1.185749 -0.056303 -0.343409 -0.285630 -0.374096 0.130722 0.037349 0.577438 0.251456 -0.279244 -0.348099 -0.297235 -0.092704 -0.301937 0.354140 -0.313855 -0.372359 0.065788 -0.267553 -0.158373 0.192400 -0.260543 -0.023510 -0.043367 0.940608 -0.140500 -0.000000 -0.111702 -0.009767 -0.369262 -0.127964 -0.197065 0.625197 0.165313 -0.115400 0.512249 0.627602 -0.117195 -0.002095 -0.300432 -0.374296 -0.366908 1.384706 -0.143056 -0.105184 -0.000000 0.258641 -0.369695 0.462185 -0.085044 -0.000000 -0.349592 -0.369304 -0.125393 0.231402 0.059901 -0.339504 -0.366358 0.277490 0.615611 -0.142484 -0.258057 -0.353778 0.270122 0.675119 -0.034393 1.623849 0.531024 1.520159 -0.208406 -0.020651 0.340055 -0.017944 -0.251129 0.100445 -0.243195 -0.366535 0.304994 -0.190456 -0.000000 1.560253 -0.170723 1.876832 -0.236901 1.147654 -0.030443 -0.335719 -0.231472 0.832267 -0.251812 0.173139 0.277327 -0.291251 -0.158537 -0.173458 0.883858 0.373644 0.031490 -0.014528 1.299525 -0.291759 0.512567 -0.176638 0.171949 -0.370697 1.118455 -0.274257 0.932934 0.338546 -0.367790 -0.226951 0.994302 -0.371637 -0.187022 0.192219 -0.362956 -0.339714 -0.011601 1.152090 -0.038030 0.256049 -0.267961 -0.283527 -0.000000 -0.372864 0.939549 0.581824 0.296811 -0.319830 -0.368415 -0.304415 -0.315848 0.926675 -0.000000 -0.150257 0.132966 0.388454 0.086585 0.287580 0.029440 0.489647 -0.331186 -0.362391 -0.315962 0.106886 -0.142415 -0.346409 -0.000000 -0.025328 -0.362296 0.642829 -0.222247 -0.257982 -0.264244 0.351465 0.035982 0.044297 -0.357779 -0.307949 1.264458 2.151624 1.100246 1.250383 -0.158189 1.065098 1.033960 -0.035196 -0.126293 0.442521 0.510706 -0.266859 -0.000000 -0.315524 -0.356934 -0.374315 -0.296549 0.470452 -0.349582 0.489689 0.714009 0.340891 0.254655 -0.292112 -0.343991 -0.374804 0.911143 0.304568 -0.258124 -0.000000 -0.172354 -0.360670 -0.245742 0.722101 0.426516 0.247256 -0.123107 -0.370158 0.330998 0.179385 -0.211047 -0.231684 0.211812 0.172235 -0.092846 0.646304 0.840956 0.612305 -0.249034 1.251980 -0.337399 -0.243072 0.212444 1.030017 -0.367093 -0.118192 0.097616 0.095499 -0.334172 -0.257267 0.303457 0.577598 0.619894 0.272026 -0.000000 -0.263152 -0.294186 -0.027608 1.901332 0.504324 0.124322 0.542341 -0.227442 -0.190705 -0.316545 0.298083 1.086621 0.245427 -0.361899 -0.000000 -0.153965 0.930369 -0.170681 -0.018823 -0.370673 0.136081 -0.257071 -0.108130 -0.013602 -0.340939 0.479296 -0.351448 -0.350976 -0.184413 1.007653 -0.137175 0.660923 -0.041111 0.125614 0.302229 -0.354386 0.971684 0.511665 -0.084981 -0.095671 -0.373802 -0.000000 -0.374238 -0.367413 -0.267296 1.650970 -0.159671 -0.034516 -0.243974 -0.006104 1.048991 0.903904 -0.268367 1.308174 -0.240137 -0.200642 0.285815 -0.222050 -0.341283 -0.364391 0.041643 -0.210552 1.193148 -0.002654 -0.318077 0.334700 -0.373548 1.412613 -0.281957 -0.000000 1.698424 -0.374953 0.006239 0.974996 0.983011 -0.348208 -0.374395 0.038422 1.049285 -0.281957 -0.359883 -0.268336 0.253683 -0.294155 -0.374999 -0.346803 -0.374538 2.059301 -0.094402 -0.000000 0.169609 -0.302031 -0.010788 0.707580 0.282777 0.386340 -0.374999 -0.180486 -0.067388 0.165172 -0.351297 -0.025505 0.431736 -0.071109 -0.357268 -0.148183 -0.283159 -0.000000 -0.295084 0.567788 -0.305107 -0.042950 0.330832 0.025284 1.041831 -0.179616 0.146365 0.801492 -0.174080 -0.317035 -0.135763 0.842999 0.361851 0.249094 -0.362960 -0.349199 -0.194464 0.193991 -0.292477 -0.239088 -0.099691 -0.000000 -0.229501 -0.267935 -0.345840 -0.111953 1.120047 0.230942 0.316887 -0.285885 -0.305537 0.620591 0.145131 1.122814 -0.000000 -0.209269 -0.236368 1.002283 -0.185742 -0.001729 -0.354330 0.421286 -0.064657 -0.000000 -0.315300 0.666594 -0.171725 -0.108373 0.092494 -0.334811 0.067285 -0.260465 -0.000000 -0.186349 0.743519 -0.164415 -0.199798 -0.361454 0.113818 -0.000000 0.512331 0.509542 -0.100333 0.212945 -0.356606 0.480310 -0.349295 -0.233251 0.413112 0.555208 -0.365773 -0.356879 -0.256053 0.048909 0.016996 -0.123281 -0.199702 -0.273480 -0.161980 -0.128137 -0.326022 -0.000000 -0.000000 -0.223720 -0.324116 -0.000000 -0.316059 -0.235316 -0.373632 -0.000000 0.376956 -0.336469 0.083681 -0.213113 0.085391 -0.217925 0.069551 0.045083 -0.346276 0.181727 -0.327014 -0.154259 -0.326594 -0.229667 -0.274476 0.281398 -0.068180 -0.228229 -0.371905 0.789097 1.032101 0.564190 -0.043883 -0.104481 -0.026159 1.262515 -0.019030 -0.000000 -0.374823 -0.000000 -0.374995 0.379710 -0.373822 -0.332297 0.818115 0.014878 -0.068282 -0.344552 0.278924 -0.000000 -0.323018 1.191326 0.407229 -0.350272 0.843384"
new_vector_array = np.fromstring(new_vector_data, sep=' ')
class_labels = ['L', 'C', 'None']
# cluster_file = 'L_vectors_001.txt'
class_index, class_label = find_similar_class(new_vector_array, file_paths, class_labels)
print('The new feature vector belongs to class:', class_label)

# def visualize_cluster(cluster_file):
#     # Load the cluster data from the text file
#     cluster_data = np.loadtxt(cluster_file)

#     # Extract the nodes
#     # nodes = cluster_data[:-1]  # Exclude the last row which represents the centroid
#     # Compute centroid from the cluster data by taking the mean
#     centroid_data = np.mean(cluster_data, axis=0)  # Compute the centroid
#     # Compute the median vector of the cluster data
#     # centroid_data = np.median(cluster_data, axis=0)

#     # Compute the cosine similarity for each vector in cluster_data
#     nodes_attribute = [np.dot(cluster_data[i], centroid_data) / (np.linalg.norm(cluster_data[i]) * np.linalg.norm(centroid_data)) for i in range(len(cluster_data))]

#     # Convert the list to a numpy array
#     nodes_attribute = np.array(nodes_attribute)

#     # Create a random centroid position within a 5x5 area
#     centroid = (np.random.uniform(0, 5), np.random.uniform(0, 5))
#     centroid = (0.0, 0.0)

#     # Create a list of nodes positions
#     nodes = []
#     for i in range(len(nodes_attribute)):
#         # Compute the norm of the nodes_attribute value
#         norm = nodes_attribute[i]
#         # Generate a random angle between 0 and 2*pi
#         angle = np.random.uniform(0, 2*np.pi)
#         # Compute the x and y coordinates relative to the centroid
#         x = centroid[0] + norm * np.cos(angle)
#         y = centroid[1] + norm * np.sin(angle)
#         # Add the (x, y) position to the list of nodes
#         nodes.append((x, y))
    

#     # Convert the list of nodes to a numpy array
#     nodes = np.array(nodes)

#     # Plot the nodes
#     plt.scatter(nodes[:, 0], nodes[:, 1], s=50, c='blue', label='Nodes', marker='o')

#     # Plot the centroid
#     plt.scatter(centroid[0], centroid[1], s=100, c='red', label='Centroid', marker='o')

#     # Plot the unitary circle
#     theta = np.linspace(0, 2*np.pi, 100)
#     x_circle = np.cos(theta)
#     y_circle = np.sin(theta)
#     plt.plot(x_circle, y_circle, color='black', linestyle='--', label='Unitary Circle')

#     # Add labels and legend
#     plt.xlabel('Distance component in x', fontsize=14)
#     plt.ylabel('Distance component in y', fontsize=14)
#     # # Remove ticks labels
#     # plt.xticks([])
#     # plt.yticks([])
#     # Set the ticks to show by every integer unit
#     plt.xticks(np.arange(int(min(nodes[:, 0])) - 1, int(max(nodes[:, 0])) + 2, 1), fontsize=11)
#     plt.yticks(np.arange(int(min(nodes[:, 1])) - 1, int(max(nodes[:, 1])) + 2, 1), fontsize=11)
#     # # Set the ticks to show by every integer unit
#     # plt.xticks(np.arange(int(min(nodes[:, 0])) - 1, int(max(nodes[:, 0])) + 2, 1))
#     # plt.yticks(np.arange(int(min(nodes[:, 1])) - 1, int(max(nodes[:, 1])) + 2, 1))
#     plt.grid(True, color='gray', alpha=0.3)
#     # legend and title
#     legend = plt.legend(fontsize='12', loc='upper left')#loc ='upper right')
#     # Get the bounding box of the legend relative to the figure
#     bbox = legend.get_window_extent().transformed(plt.gca().transData.inverted())
#     # Calculate width and height of the legend box
#     x_inf_lim, x_sup_lim = plt.xlim()#ax.get_xlim()
#     y_inf_lim, y_sup_lim = plt.ylim()#ax.get_ylim()
#     legend_width = bbox.width/(x_sup_lim-x_inf_lim)
#     legend_height = bbox.height/(y_sup_lim-y_inf_lim)
#     print('legend_width =', legend_width)
#     print('legend_height =', legend_height)
#     # The y position for centering the legend
#     legend_y_pos = 0.5*(1+legend_height)
#     print ('legend_y_pos =',legend_y_pos)
#     legend.set_bbox_to_anchor((1.05, legend_y_pos))

#     # Adjust the figure size with rcParams
#     plt.rcParams['figure.figsize'] = [9, 6]
#     # Adjust the figure margins
#     plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)

#     # Show the plot
#     plt.show()
