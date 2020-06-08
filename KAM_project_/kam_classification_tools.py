""" 
Authors: Luke A.
Software: Python 3.7.3, 64-bit
Projec: ARM

Required input file's columns
1) report_id
2) kam_id
3) auditor
4) company 
5) kam_acct_topic

* sic_code and siccode contain different missing observations... so I combined them later in the code =*=
* hopefully, this won't happen in US CAM data.
6) sic_code   
7) siccode

8) partner
9) kam_description
10) kam_addressing

#-------------------------------------------------------------------------------------------------------------
Version Control
#-------------------------------------------------------------------------------------------------------------
nltk==3.4.1
re==2.2.1
numpy==1.16.4
pandas==0.24.2
matplotlib==2.2.4
scipy==1.3.0

"""

# load all external packages
import nltk,time, glob, pickle, re, os, numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd, numpy as np, string
from sklearn.metrics import silhouette_samples, silhouette_score
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans, DBSCAN
from sklearn.pipeline import Pipeline
import matplotlib.cm as cm
from scipy.spatial.distance import cosine as csd
from nltk.stem import PorterStemmer
from scipy.sparse import hstack
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from mpl_toolkits import mplot3d
import webbrowser as wb
from matplotlib import interactive

# load all special characters
from KAM_project_4.special_char import *

interactive(True) # this is for TSNE plots
unwanted = set(nltk.corpus.stopwords.words('english') + list(string.punctuation) + stuff) # load all unwanted characters
stemmer = PorterStemmer() # Use Porter Stemmer for stematiztion

def transform_and_train(file,my_item,e, my_year='', my_auditor='',my_topic='',my_sic='',min_s=2,length_feature=False, output =False, music=False):
    """
	
	Description for the `transform_and_train` function:
	====================================================================================================
	Inputs:
	====================================================================================================
		file            : xlsx filename.
		my_item         : kam item. Currently, it is either kam_description or kam_addressing.
		e               : The maximum distance between two samples for one to be considered as in the 
                                neighborhood of the other. This is the most important DBSCAN parameter to 
                                choose appropriately for your data set and distance function.
		my_year         : year (int). The default value is ''.
                my_auditor      : auditor name (str). The default value is ''.
                my_topic        : accounting topic (str). The default value is ''.
                my_sic          : my first sic (int). The default value is ''.
		min_s	        : The number of samples in a neighborhood for a point to be considered 
				as a core point for DBSCAN. The default value is 2.
		length_feature 	: A boolean parameter to specify whether the length of sentences should be 
                                used as a feature or not. The default value is False.
                output          : default = False, export an xlsx file to the Ouputs directory if True.
    
	====================================================================================================
	Outputs:
	====================================================================================================
		df_sent         : The initial output from the 1st step of DBSCAN. This is required for the 
		                multi-DBSCAN step.
	
	====================================================================================================
	"""
    
    # Load the saved python objects if they exist. This is a very smart trick from Josh Smith to speed up the runtime. The idea is that 
    # all text preprocessing steps will be run just once and saved as *.p file. However, I added a housekeeping step into another script.
    # This is important, since if we don't remove the *.p files before running a new study with new parameters. It will use the results from
    # the previous study as a starting point and we will end up with a misleading result!

    # psst** by the way,  you may add parameter 'music =True' into the function for better performance.

    # You will see this trick down below too in another function. For here, it is for loading the original data if the *.p does not exist.
    if os.path.isfile('../Intermediates/tokenized_data.p'):
        testdata = None
        
        # Load additional information such as auditors, issuers, and etc to be added to the final output
        additional_info = open('..\\Intermediates\\add_info.p', 'rb')
        add_info = pickle.load(additional_info)
        additional_info.close()
    else:
        # if *.p file does not exist, go through  the text preprocessing steps. Load the data from beginning.
        testdata, add_info = load_sheet(file,my_item,my_year,my_auditor,my_topic,my_sic) 


    df2, original_sent = preprocess(testdata,my_item) # preprocessing steps

    # Extract 1 to 6 grams features and calculate the tfidf statistics for all sentences. We also remove all unwanted characters.                        
    vectorizer = TfidfVectorizer(ngram_range=(1,6), 
                             stop_words=unwanted,
                             min_df=1, max_df=1.0,
                             norm='l2', use_idf=True)

    # Extract for real
    X = vectorizer.fit_transform(df2['sentence'])

    # experimenting with adding length of the sentence as another feature
    if length_feature == True :
        add_w_count = pd.DataFrame(df2.sentence.apply(lambda x: len(x.split(' '))))
        X = hstack((X,add_w_count))	


    print('Training...\n')
    # Use DBSCAN to classify sentences into groups    
    clusterer = DBSCAN(e,min_samples=min_s)
    clusterer.fit(X)
    # Keep the group labels and add them to the final output
    cluster_group = clusterer.labels_
    df2['cluster_group'] = cluster_group
    df2 = df2[['cluster_group', 'id']]

    # Add the original sentences back to the final output
    df3 = pd.merge(df2,original_sent, on='id',how='left')

    df3.drop(['id'],axis = 1,inplace = True) # drop the id column

    # Add the additional information we need from add_info
    df3 = pd.merge(df3,add_info,on='kam_id',how='left')

    df3['group_count'] = df3.groupby('cluster_group')['cluster_group'].transform('count')
    n_noises = len(df3[df3.cluster_group == -1])
    
    # Create sentence id
    df3['sent_no'] = df3.groupby(['kam_id']).cumcount()+1
    df3['sent_no'] = df3['sent_no'].apply(str)
    df3['sent_id']  = df3[['kam_id','sent_no']].apply(lambda x: '_s'.join(x),axis=1)
    
    # Get 2 digits of SIC code
    df3['sic_2'] = df3.sic_code.apply(lambda x: str(x)[0:2])
    df3['sic_2'] = df3['sic_2'].astype(int)
    df3 = df3[['sent_id','kam_id','cluster_group','group_count','year','auditor','company','kam_acct_topic','partner','sic_code','sic_2','1st_digit_sic','sentence']]
    
    
    # Argument for generating the excel file.
    if output == True:
        df3.sort_values('cluster_group',ascending=False).to_excel('..\\Intermediates\\'+my_item+f'_eps_{e}_{n_noises}_1st_iter.xlsx',index=False)
        print('\nfilename: '+my_item+f'_eps_{e}_{n_noises}_1st_iter.xlsx')    
    
    # Don't mind me. I need some music.
    if music ==True:
        wb.open_new('https://www.youtube.com/watch?v=oO7Y8NsnkRg')
        print(
                '     Friend:  Hey! How are you man?\n \
    Me    :  PA PA PA PA YAAAAAAAAAAA ಠ◡ಠ\n \
    Friend:  ┌( ಠ_ಠ)┘ ┌( ಠ_ಠ)┘ ┌( ಠ_ಠ)┘')
    print('\nDone')
    return df3


def load_sheet(input_file,my_item, my_year, my_auditor, my_topic, my_sic):
    xl1 = pd.ExcelFile("..\\Inputs\\"+input_file)
    data = xl1.parse('Sheet1', header=0)
    data.drop_duplicates(subset='kam_id',inplace=True) # drop duplicated observations from the start
    
    # For the UK KAM data, there are 2 names for KPMG. So we combined them to 'KPMG LLP'
    data['auditor'] = data['auditor'].replace({'KPMG AUDIT PLC':'KPMG LLP'})

    # There is human error on data collection for years. To avoid this, we extract the year from the report id directly!
    # ex there is 2014 in the observation for 2013, or the other way around.
    data['year']    = data.report_id.apply(lambda x:re.findall('\d{4}',str(x))[0])
    data['year']    = data['year'].astype(int)

    # There are two sic code columns which are not the same!!! So I combine them.
    data.sic_code = data[['sic_code','siccode']].apply(lambda x: np.nanmax(x),axis=1)

    # Create a column for the first digit of sic code 
    data.sic_code = data.sic_code.fillna(0).astype(int)
    data['sic_1'] = data.sic_code.apply(lambda x: str(x)[0])
    data['sic_1'] = data['sic_1'].astype(int)

    # Settings for filtering by year, auditor, topic, sic or any combination.
    if my_year:
	    data = data[data['year']==my_year]

    if my_auditor:
	    data = data[data['auditor']==my_auditor]

    if my_topic:
	    data = data[data['kam_acct_topic']==my_topic]

    if my_sic:
        data = data[data['sic_1']==my_sic]

    data2 = data[['kam_id', my_item]]

    # create a dataframe for additional information to be kept
    add_info = data[['kam_id','year','auditor','kam_acct_topic','sic_code','company','partner',my_item]].copy()
    add_info.sic_code = add_info.sic_code.fillna(0).astype(int)
    add_info= pd.concat([add_info,add_info.sic_code.apply(lambda x: str(x)[0]).rename('1st_digit_sic',axis=1)],axis=1)
    
    out3 = open('..\\Intermediates\\add_info.p','wb')
    pickle.dump(add_info,out3)
    out3.close()
 

    return data2[data2[my_item].notnull()].reset_index(drop=True), add_info


def preprocess(testdata, my_item):
    df2 = pd.DataFrame(columns=['kam_id', 'sentence', 'id'])

    print('Text Preprocessing...')
    
    # Be very careful about this, ask yourself if you need to remove the *.p files or not for each study
    try:
        if not testdata:
            tokenized_sents = open('..\\Intermediates\\tokenized_data.p', 'rb')
            orig_sent = open('..\\Intermediates\\original_sents.p', 'rb')
            return pickle.load(tokenized_sents),pickle.load(orig_sent)
            tokenized_sents.close()
            orig_sent.close()

    except:
        pass
    
    unmodified_sentences = pd.DataFrame(columns=['kam_id', 'sentence', 'id'])
    # All the text preprocessing steps are happening here. Sentence tokenization, word tokenization, replacing contractions, stematization.
    for row in testdata.iterrows():
        paragraph = row[1][my_item]
        kam_id = row[1]['kam_id']
        sentences = nltk.sent_tokenize(paragraph)

        for sentence in sentences:
            # keep original text
            unmodified_sentences = unmodified_sentences.append(
                {'kam_id': kam_id, 'sentence': sentence}, ignore_index=True)
        sentences = [replace_contractions(x) for x in sentences]
        sentences = [' '.join([stemmer.stem(i) for i in word_tokenize(x)]).rstrip() for x in sentences] # lemmatization has to be done at the word level, not sentence level!
        for sentence in sentences:
            df2 = df2.append({'kam_id': kam_id, 'sentence': sentence},
                             ignore_index=True)
    df2['id'] = df2.index
    
    out1 = open('..\\Intermediates\\tokenized_data.p', 'wb+')
    pickle.dump(df2, out1)
    out1.close()
    
    unmodified_sentences['id'] = df2.index
    out2 = open('..\\Intermediates\\original_sents.p', 'wb+')
    pickle.dump(unmodified_sentences, out2)
    out2.close()
    
    return df2, unmodified_sentences


def replace_contractions(text): # Function for replacing special characters or contractions
    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    def replace(match):
        return contraction_dict[match.group(0)]
    return contraction_re.sub(replace, text)


def seqNoisesClassfication(e,my_item,df,my_gram=(1,6),out_path='',length_feature=False,output=False):
    """
	Description for the `seqNoisesClassfication` function:
	====================================================================================================
	Inputs:
	====================================================================================================
		e               : The maximum distance between two samples for one to be considered as in the 
                                neighborhood of the other. This is the most important DBSCAN parameter to 
                                choose appropriately for your data set and distance function.
		my_item         : kam item. Currently, it is either kam_description or kam_addressing.
		df              : The dataframe from the `transform_and_train` function.
		my_gram	        : n-gram to be used for the analysis. default = (1,6). 
		out_path        : The directory path for the multi-DBSCAN step's outputs.
                length_feature  : A boolean parameter to specify whether the length of sentences should be 
                                used as a feature or not. The default value is False.
                output          : default = False, export an xlsx file to the Ouputs directory if True.
    
	====================================================================================================
	Outputs:
	====================================================================================================
		df_sent_final   : The output from the multi-DBSCAN step.
	
	====================================================================================================
	"""
    # This function is for running DBSCAN iteratively until it exhaust all possible grouping possible.
    df['new_group'] = ''
    list_col = df.columns.tolist()[:-2] +['new_group','sentence']
    df = df[list_col]
    cut_off_sent = 0.1*len(df)
    
    count = 1
    noise_count = [0,df.group_count.max()]
    
    # Keep running DBSCAN while the number of unique sentences can still be reduced. 
    while (df.group_count.max() > cut_off_sent) and (df.group_count.max() != noise_count[count-1]):    
    
        prev_max_sent = df.cluster_group.max()
        cond_cut = df.cluster_group==-1
        df3 = df[cond_cut].copy()

        text = df3.sentence.apply(lambda x: stemmer.stem(replace_contractions(x).lower().lstrip()))
        vectorizer = TfidfVectorizer(ngram_range=my_gram,stop_words=unwanted,
                                     min_df=1, max_df=1.0, norm='l2', use_idf=True)
        X = vectorizer.fit_transform(text)
        df4 = df3.copy()
        if length_feature == True :
            add_w_count = pd.DataFrame(df3.sentence.apply(lambda x: len(x.split(' '))))
            X = hstack((X,add_w_count))
        
        clusterer = DBSCAN(eps=e,min_samples=2)
        clusterer.fit(X)
        cluster_group = clusterer.labels_
        df3['new_group'] = cluster_group
        df3.loc[df3.new_group >=0,'new_group'] = df3.loc[df3.new_group >=0,'new_group']+1+prev_max_sent
        df3.cluster_group = df3.new_group
        df3.group_count = df3.groupby('cluster_group')['cluster_group'].transform('count')
        df = df[~cond_cut].copy().append(df3, ignore_index=True)
        count +=1
        noise_count.append(df.group_count.max())		
        n_noises = len(df[df.cluster_group == -1])
    
    opt_inter = noise_count[1:-1]
    
    print('\nIterations corresponding to n-gram =',my_gram)
    print('Number of iterations needed:',len(opt_inter))    
    print('Number of sentences in the noise group in each iteration:',opt_inter)
    print('-----------------------------------------------------------------------------------')
    df.drop(['new_group'],axis=1,inplace=True)

    if output == True:
        df.to_excel('..\\Outputs\\'+out_path+'\\'+my_item+f'_sentence_eps_{e}_{n_noises}.xlsx',index=False)
        print('\nOutput Filename: '+my_item+f'_sentence_eps_{e}_{n_noises}.xlsx')
        print('-----------------------------------------------------------------------------------')
    
    return df
  
def kamClassifcation(df,my_item,excel_file, es = [1.0],out_path='',output=False):
    """
	Description for the `kamClassifcation` function:
	====================================================================================================
	Inputs:
	====================================================================================================
		df              : The dataframe from the `seqNoisesClassfication` function.
		my_item         : kam item. Currently, it is either kam_description or kam_addressing.
		excel_file            : xlsx filename.
		my_gram	        : n-gram to be used for the analysis. default = (1,6).
                es              : A list of epsilon values.  
		out_path        : The directory path for the kam_report DBSCAN step's output.
                output          : default = False, export an xlsx file to the Ouputs directory if True.
    
	====================================================================================================
	Outputs:
	====================================================================================================
		final_kam_group : The output from the kam_report DBSCAN step.
                label_df        : The labels from the kam_report DBSCAN step.
	
	====================================================================================================
	"""
    # Before we use DBSCAN at the sentence level, now we use it on the KAM component level (loosely speaking, at the paragraph level).
    # Preparation for kam_DBSCAN
    unique_df = df[df.cluster_group==-1].copy()
    unique_df.reset_index(drop=True,inplace=True)
    unique_df.reset_index(inplace=True)
    unique_df.drop(['cluster_group'],axis=1,inplace=True)
    unique_df.rename(columns={'index': 'cluster_group'},inplace =True)
    unique_df.cluster_group = unique_df.cluster_group +1 + df.cluster_group.max()
    
    new = df[df.cluster_group!=-1].copy()
    new = new.append(unique_df, sort=False,ignore_index=True)
    
    new = new[['kam_id','cluster_group']].copy()
    new['label'] = new.cluster_group.apply(lambda x: 'label_'+str(x))
    label_df = pd.DataFrame(pd.pivot_table(pd.DataFrame(new.groupby(['kam_id', 'label']).size()),columns='label',index='kam_id').to_records()).fillna(0)
    label_df.columns = [hdr.replace("(0, '",'').replace("')",'') for hdr in label_df.columns]
    
    for e in es:
        # First DBSCAN for kam_doc
        X2 =label_df[label_df.columns.to_list()[1:]]
        clusterer = DBSCAN(eps=e,min_samples=2)
        clusterer.fit(X2)
        cluster_group = clusterer.labels_
        
        kam_group_df = pd.DataFrame({'kam_id':label_df.kam_id})
        kam_group_df['kam_group'] = cluster_group
#        print('kam_group_df',kam_group_df.shape,'\n')

        original_data = pd.read_excel('..\\Inputs\\'+excel_file)
        original_data.drop_duplicates(subset='kam_id',inplace=True)
        original_data['year']    = original_data.report_id.apply(lambda x:re.findall('\d{4}',str(x))[0])
        original_data['year']    = original_data['year'].astype(int)
        original_data.sic_code = original_data[['sic_code','siccode']].apply(lambda x: np.nanmax(x),axis=1)
        original_data = original_data[['kam_id','year','auditor','company','partner','kam_acct_topic','sic_code',my_item]]	
        original_data.rename(columns={my_item : 'text'},inplace =True)
        
#        print('original_data',original_data.shape,'\n')
        original_data = original_data[original_data['text'].notnull()].reset_index(drop=True)
#        print('original_data_2',original_data.shape,'\n')
        
        final_kam_group = pd.merge(original_data,kam_group_df,on='kam_id',how='inner')
#        print('after merge',final_kam_group.shape,'\n')
        
        final_kam_group['group_count'] = final_kam_group.groupby('kam_group')['kam_group'].transform('count')
        final_kam_group['year'] = final_kam_group.year.fillna(0).astype(int)
        final_kam_group['sic_code'] = final_kam_group.sic_code.fillna(0).astype(int)
        n_noises = len(final_kam_group[final_kam_group.kam_group == -1])
        
        if output == True:
            print('\nOutput Filename: ', my_item+f'_report_{e}_{n_noises}.xlsx')
            final_kam_group.to_excel('..\\Outputs\\'+out_path+'\\'+my_item+f'_report_{e}_{n_noises}.xlsx',index=False)

    return final_kam_group, label_df, original_data

def TSNE_2Dplot(final_kam_group, label_df ,my_rate,size_tuple):
    """
	Description for the `TSNE_2Dplot` function:
	====================================================================================================
	Inputs:
	====================================================================================================
		final_kam_group : The output from the kam_report DBSCAN step.
                label_df        : The labels from the kam_report DBSCAN step.
            my_rate         : The learning rate for t-SNE is usually in the range [10.0, 1000.0].
            size_tuple      : Figure size.
    
	====================================================================================================
	Outputs:
	====================================================================================================
        A 2 dimensional plot.
	
	====================================================================================================
	"""

    label_df = label_df[label_df.kam_id.isin(final_kam_group.kam_id)]
    elim = final_kam_group.kam_id[final_kam_group.kam_id.duplicated()].index

    if len(elim)==0:
    	kam_group_labels = final_kam_group.kam_acct_topic
    else:
        kam_group_labels = final_kam_group.iloc[final_kam_group.index != elim.values[0]].kam_group

    kam_sent_labels = label_df.loc[:, label_df.columns !='kam_id']	
	
	# Create a TSNE instance: model
    model = TSNE(learning_rate = my_rate)
    tsne_features = model.fit_transform(kam_sent_labels)

    xs = tsne_features[:,0]
    ys = tsne_features[:,1]
	
    figure(num=None, figsize= size_tuple, dpi=80, facecolor='w', edgecolor='k')
    plt.scatter(xs,ys)
    plt.show()
 
def TSNE_3Dplot(final_kam_group, label_df ,my_rate,size_tuple):
    """
	Description for the `TSNE_3Dplot` function:
	====================================================================================================
	Inputs:
	====================================================================================================
		final_kam_group : The output from the kam_report DBSCAN step.
                label_df        : The labels from the kam_report DBSCAN step.
            my_rate         : The learning rate for t-SNE is usually in the range [10.0, 1000.0].
            size_tuple      : Figure size.
    
	====================================================================================================
	Outputs:
	====================================================================================================
        A 3 dimensional plot.
	
	====================================================================================================
	"""

    label_df = label_df[label_df.kam_id.isin(final_kam_group.kam_id)]
    elim = final_kam_group.kam_id[final_kam_group.kam_id.duplicated()].index    
    if len(elim)==0:
    	kam_group_labels = final_kam_group.kam_acct_topic
    else:
    	kam_group_labels = final_kam_group.iloc[final_kam_group.index != elim.values[0]].kam_group  
    kam_sent_labels = label_df.loc[:, label_df.columns !='kam_id']	

    # Create a TSNE instance: model
    model = TSNE(n_components=3, learning_rate = my_rate)
    tsne_features = model.fit_transform(kam_sent_labels)    
    xs = tsne_features[:,0]
    ys = tsne_features[:,1]
    zs = tsne_features[:,2]

    figure(num=None, figsize= size_tuple, dpi=80, facecolor='w', edgecolor='k')
    ax = plt.axes(projection='3d')
    ax.scatter3D(xs, ys, zs);
    plt.show() 
    
def QC_uniqueSentences_DB(df_sent_final,my_item,by_chunk=True,output=False):
    """
	Description for the `QC_uniqueSentences_DB` function:
	====================================================================================================
	Inputs:
	====================================================================================================
            df_sent_final       : The dataframe from the `seqNoisesClassfication` function.
	    my_item             : kam item. Currently, it is either kam_description or kam_addressing.
            by_chunk            : default = True = QC by chunk. QC by sentence if False.
            output              : default = False, export an xlsx file to the Ouputs directory if True.
    
	====================================================================================================
	Outputs:
	====================================================================================================
	    df_check                : A QC report.
	
	====================================================================================================
	"""
    # For checking results
    if by_chunk == True:
        # Check all at once, run DBSCAN on all unique sentences with grouped sentences to see if they can be regrouped.
        df = df_sent_final.copy()
        text = df.sentence.apply(lambda x: ' '.join([stemmer.stem(i) for i in word_tokenize(replace_contractions(x))]).rstrip())
        vectorizer = TfidfVectorizer(ngram_range=(1,6),stop_words=unwanted, min_df=1, max_df=1.0, norm='l2', use_idf=True)
        X = vectorizer.fit_transform(text)
        clusterer = DBSCAN(eps=0.99,min_samples=2)
        clusterer.fit(X)
        df['new_group'] = clusterer.labels_
        df_check = df[['sent_id','cluster_group','new_group','sentence']].copy()
        df_check['flag'] = ''
        df_check['flag'] = df_check.cluster_group != df_check.new_group
        df_check = df_check[df_check.cluster_group==-1].copy()
        print("Number of unique sentences got regrouped :", len(df_check[df_check.flag==True]))
        
        if output == True:
            print('\nOutput Filename: ', my_item+f'_QC_uniqueSentences_DB_by_chunk.xlsx')
            df_check.to_excel('..\\Outputs\\'+my_item+f'_QC_uniqueSentences_by_chunk.xlsx',index=False)

        return df_check
    else:
        # Run DBSCAN on one unique sentence and other classified sentences, one at a time!! => take a while to run.
        unique_sent_df = df_sent_final[df_sent_final.cluster_group==-1].copy()
        unique_sent_df.reset_index(drop=True,inplace=True) 
        print('Number of unique sentences : ',unique_sent_df.shape[0])

        grouped_sent_df = df_sent_final[df_sent_final.cluster_group!=-1].copy()
        grouped_sent_df.reset_index(drop=True,inplace=True)
        print('Number of grouped sentences: ',grouped_sent_df.shape[0])

        check_unique_sent = unique_sent_df[['sent_id','cluster_group','sentence']].copy()
        check_unique_sent['new_group'] = ''
        for idx in unique_sent_df.index:
            df = grouped_sent_df.append(unique_sent_df.iloc[idx]).copy()
            text = df.sentence.apply(lambda x: ' '.join([stemmer.stem(i) for i in word_tokenize(replace_contractions(x))]).rstrip())
            vectorizer = TfidfVectorizer(ngram_range=(1,6),stop_words=unwanted, min_df=1, max_df=1.0, norm='l2', use_idf=True)
            X = vectorizer.fit_transform(text)
            clusterer = DBSCAN(eps=0.99,min_samples=2)
            clusterer.fit(X)
            check_unique_sent.iloc[idx,3] = clusterer.labels_[-1]
        
        if output == True:
            print('\nOutput Filename: ', my_item+f'_QC_uniqueSentences_DB_by_sent.xlsx')
            check_unique_sent.to_excel('..\\Outputs\\'+my_item+f'_QC_uniqueSentences_by_sent.xlsx',index=False)
    
        return check_unique_sent

def QC_uniqueSentences_cosine(df_sent_final,my_item,min_cosine = 0.1,output=False, my_col = 'maroon'):
    """
	Description for the `QC_uniqueSentences_cosine` function:
	====================================================================================================
	Inputs:
	====================================================================================================
            df_sent_final   : The dataframe from the `seqNoisesClassfication` function.
	    my_item         : kam item. Currently, it is either kam_description or kam_addressing.
            min_cosine      : The minimum value of cosine score to be considered. default = 0.1.
            output          : default = False, export an xlsx file to the Ouputs directory if True.
            my_col          : color for the histogram. default = 'maroon'
    
	====================================================================================================
	Outputs:
	====================================================================================================
	    matched_sent        : A QC report.
            histogram
	
	====================================================================================================
	"""
    # For checking results with cosine similarity scores.
    df_check = df_sent_final.copy()
    df_check = df_check[df_check.cluster_group==-1]
    text = df_check.sentence.apply(lambda x: ' '.join([stemmer.stem(i) for i in word_tokenize(replace_contractions(x))]).rstrip())
    vectorizer = TfidfVectorizer(ngram_range=(1,6),stop_words=unwanted,min_df=1, max_df=1.0, norm='l2', use_idf=True)
    X = vectorizer.fit_transform(text)
    cosine = cosine_similarity(X=X, Y=X, dense_output=True)
    cosine_mat = pd.DataFrame(cosine,columns=df_check.sent_id,index=df_check.sent_id)
    cosine_mat = cosine_mat.rename_axis('sent_id0')
    cosine_mat = cosine_mat.rename_axis('sent_id1',axis='columns')
    upper_tri_index = np.triu(np.ones(cosine_mat.shape[0]),1).astype(np.bool)
    cosine_mat_2 = cosine_mat.where(upper_tri_index)
    cosine_mat_2 = cosine_mat_2.stack().reset_index()
    cosine_mat_2.rename({0:'cosine_score'},axis=1,inplace=True)

    # Max cosine_score
    print('Maximum value of Cosine Similarity Score:', cosine_mat_2.cosine_score.max())

    match_sent = df_sent_final[['sent_id','auditor','company','sentence']]
    matched_sent = pd.merge(cosine_mat_2,match_sent,left_on='sent_id0',right_on='sent_id',how='left')
    matched_sent.drop(['sent_id'],axis=1,inplace=True)
    matched_sent.rename({'sentence':'sentence_0',
                         'auditor':'auditor_0',
                         'company':'issuer_0'},inplace=True,axis=1)

    matched_sent = pd.merge(matched_sent,match_sent,left_on='sent_id1',right_on='sent_id',how='left')
    matched_sent.drop(['sent_id'],axis=1,inplace=True)
    matched_sent.rename({'sentence':'sentence_1',
                         'auditor':'auditor_1',
                         'company':'issuer_1'},inplace=True,axis=1)
    matched_sent = matched_sent[['sent_id0','sent_id1','auditor_0','auditor_1','issuer_0','issuer_1','sentence_0','sentence_1','cosine_score']]

    matched_sent2 = matched_sent.loc[matched_sent.cosine_score>min_cosine].copy()

    fig, ax = plt.subplots()
    matched_sent2.cosine_score.plot.hist(bins=100, color=my_col)
    ax.set_xlabel("Cosine Scores")
    fig.savefig('..\\Outputs\\'+my_item+'_QC_uniqueSentences_cosine.jpeg')

    if output == True:
        print('\nOutput Filename: ', my_item+f'_QC_uniqueSentences_cosine.xlsx')
        matched_sent2.to_excel('..\\Outputs\\'+my_item+f'_QC_uniqueSentences_cosine.xlsx',index=False)

    return matched_sent2



def genSummaryTable(my_item,df_sent_final,final_kam_group,group_filer = ['year'],output=False):
    """
	Description for the `genSummaryTable` function:
	====================================================================================================
	Inputs:
	====================================================================================================
	    my_item             : kam item. Currently, it is either kam_description or kam_addressing.
            df_sent_final       : The dataframe from the `seqNoisesClassfication` function.
            final_kam_group     : The output from the kam_report DBSCAN step.
            group_filer         : A list of filtered groups. default = ['year']
            output              : default = False, export an xlsx file to the Ouputs directory if True.
    
	====================================================================================================
	Outputs:
	====================================================================================================
	    sd_tb               : A summary table.
	
	====================================================================================================
	"""
    # Create frequency tables for generating graphs later. We can filter the output by any combination of variables,
    # such as auditor, topic, sic, and so on. No need to rerun by part from the beginning. Cool? 
    df_sent_final_tb      = df_sent_final.copy()
    df_sent_final_tb.year = df_sent_final_tb.year.fillna(0).astype(int)

    df_sent_final_tb['anchor'] = 'ALL'
    df_sent_final_tb.loc[df_sent_final_tb.year==0,'year'] = 'missing'
    yr_c     = df_sent_final_tb.groupby(group_filer)['kam_id'].count()
    all_yr_c = df_sent_final_tb.groupby('anchor')['kam_id'].count()
    yr_p     = df_sent_final_tb.groupby(group_filer).apply(lambda x: x[x.cluster_group!=-1].shape[0]/x.shape[0]*100)
    all_yr_p = df_sent_final_tb.groupby('anchor').apply(lambda x: x[x.cluster_group!=-1].shape[0]/x.shape[0]*100)
    sent_df = pd.DataFrame({'Type'   : 'sentence','Item'   : my_item,'Total'  :yr_c.append(all_yr_c),'Percent':yr_p.append(all_yr_p)})
    sent_df.Percent = sent_df.Percent.round(2)

    final_kam_group['anchor'] = 'ALL'
    final_kam_group.loc[final_kam_group.year==0,'year'] = 'missing'

    yr_c     = df_sent_final_tb.groupby(group_filer)['kam_id'].nunique()
    all_yr_c = df_sent_final_tb.groupby('anchor')['kam_id'].nunique()
    yr_p     = final_kam_group.groupby(group_filer).apply(lambda x: x[x.kam_group!=-1].shape[0]/x.shape[0]*100)
    all_yr_p = final_kam_group.groupby('anchor').apply(lambda x: x[x.kam_group!=-1].shape[0]/x.shape[0]*100)

    doc_df = pd.DataFrame({'Type'   : 'KAM component','Item'   : my_item,'Total'  :yr_c.append(all_yr_c),'Percent':yr_p.append(all_yr_p)})
    doc_df.Percent = doc_df.Percent.round(2)

    sd_tb = sent_df.append(doc_df)
    sd_tb.reset_index(inplace=True)
    sd_tb.rename({'index':'Filter'},axis=1,inplace=True)
    sd_tb = sd_tb[sd_tb.Filter != 'missing']
    sd_tb.set_index(['Item','Filter', 'Type'], inplace=True)
    sd_tb.sort_index(inplace=True)
    

    if output == True:
        filters = '_'.join(group_filer)
        print('\nOutput Filename: ', my_item+f'_SummaryTable_{filters}.xlsx')
        sd_tb.to_excel('..\\Outputs\\'+my_item+f'_SummaryTable_{filters}.xlsx')
    
    return sd_tb
