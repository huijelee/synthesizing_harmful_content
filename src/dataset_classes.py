import os
import re
import json
import random
import string
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict
from pathlib import Path
from glob import glob
from tqdm.std import tqdm
from filelock import FileLock
from dataclasses import dataclass, asdict

from datasets import Dataset, DatasetDict, load_dataset

FLAG_LIST = ['[engage]', '[ignore]', '[expose]', '[challenge]', '[critique]', '[mock]', '[reciprocate]']
FLAG_LIST_DEPRECATED = ['(in an engage way)', '(in an ignore way)', '(in an expose way)', '(in a challenge way)', '(in a critique way)', '(in a mock way)', '(in a reciprocate way)']
FLAG_STS = ['[overt]', '[covert]']
FLAG_TS = ['[aggression]', '[shocking]', '[endangering]', '[antipathy]', '[hypocriticism]', '[digression]']
FLAG_SRS = ['[friendly]', '[confront]']
FLAG_ORDERED = FLAG_RS = ['[engage]', '[ignore]', '[expose]', '[challenge]', '[critique]', '[mock]', '[reciprocate]']
FLAG_TROLL = ['[nontroll]', '[troll]']


convert_flag_dict = {k: v for k, v in zip(FLAG_LIST_DEPRECATED, FLAG_LIST)}
SPLIT = ['train', 'validation', 'test']


def convert_deprecated_flags(df):
    df['Flag'] = df.apply(lambda row: convert_flag_dict.get(row['Flag'], np.nan), axis=1)
    df.dropna(subset=['Flag'], inplace=True)
    return df


@dataclass
class DynamicAttributes:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


@dataclass
class RedditPost:
    author: str
    author_fullname: str
    author_created_utc: int
    created_utc: int
    retrieved_utc: int
    id: str
    subreddit: str
    subreddit_id: str
    title: str
    selftext: str
    permalink: str
    score: int
    subreddit_subscribers: int
    num_comments: int
    url: str

    @classmethod
    def from_dict(cls, data: Dict) -> 'RedditPost':
        """Create RedditPost from dictionary with default values"""
        defaults = {
            'author': '',
            'author_fullname': '',
            'author_created_utc': 0,
            'created_utc': 0,
            'retrieved_utc': 0,
            'id': '',
            'subreddit': '',
            'subreddit_id': '',
            'title': '',
            'selftext': '',
            'permalink': '',
            'score': 0,
            'subreddit_subscribers': 0,
            'num_comments': 0,
            'url': ''
        }
        # Update defaults with provided data
        defaults.update({k: v for k, v in data.items() if v is not None})
        return cls(**defaults)

    @classmethod
    def from_df_row(cls, row: pd.Series) -> 'RedditPost':
        """Create RedditPost from DataFrame row"""
        return cls.from_dict(row.to_dict())

    def to_dict(self) -> Dict:
        """Convert to dictionary with all fields"""
        return asdict(self)


@dataclass
class RedditComment:
    author: str
    author_fullname: str
    author_created_utc: int
    created_utc: int
    retrieved_on: str  # Note: this is 'retrieved_on' in RC vs 'retrieved_utc' in RS
    link_id: str
    parent_id: str
    id: str
    subreddit: str
    subreddit_id: str
    body: str
    permalink: str
    score: int
    depth: int
    path: list
    url: str
    dynamic_attributes: DynamicAttributes = None

    @classmethod
    def from_dict(cls, data: Dict) -> 'RedditComment':
        """Create RedditComment from dictionary with default values"""
        defaults = {
            'author': '',
            'author_fullname': '',
            'author_created_utc': 0,
            'created_utc': 0,
            'retrieved_on': '',
            'link_id': '',
            'parent_id': '',
            'id': '',
            'subreddit': '',
            'subreddit_id': '',
            'body': '',
            'permalink': '',
            'score': 0,
            'depth': 0,
            'path': [],
            'url': '',
            'dynamic_attributes': {},
        }

        # Update defaults with provided data
        defaults.update({k: v for k, v in data.items() if k in defaults})

        # Store additional attributes
        dynamic_attributes = {k: v for k, v in data.items() if k not in defaults}
        defaults['dynamic_attributes'] = DynamicAttributes(**dynamic_attributes)
        
        return cls(**defaults)

    @classmethod
    def from_df_row(cls, row: pd.Series) -> 'RedditComment':
        """Create RedditComment from DataFrame row"""
        return cls.from_dict(row.to_dict())

    def to_dict(self) -> Dict:
        """Convert to dictionary with all fields"""
        return asdict(self)

    def __str__(self):
        comment_dict = {
            "body": self.body,
            "id": self.id,
            "path": self.path
        }
        return json.dumps(comment_dict)



@dataclass
class Thread:
    post: RedditPost
    comments: List[RedditComment]
    max_depth: int

    def make_complete(self) -> 'Thread':
        """
        Clean thread by removing comments with broken chains.
        Returns new Thread with only valid comment paths.
        """
        if not self.comments:
            return self

        unique_comments = {comment.id: comment for comment in self.comments}.values()

        # First pass: organize comments by id
        comment_by_id = {}
        valid_comments = []

        # Build full chains and validate
        for comment in unique_comments:
            valid_chain = True
            current = comment
            chain = [current]

            # Trace back to root for each comment
            while True:
                # Root comment
                if current.parent_id == f"t3_{self.post.id}":
                    break
                # Nested comment
                elif current.parent_id.startswith('t1_'):
                    parent_id = current.parent_id[3:]

                    if parent_id not in comment_by_id:
                        valid_chain = False
                        break

                    current = comment_by_id[parent_id]
                    chain.append(current)
                else:
                    valid_chain = False
                    break

            if valid_chain:
                # Fix depth and path
                chain.reverse()  # Now from root to leaf
                # Create new comment with corrected depth and path
                comment_dict = comment.to_dict()
                comment_dict.update({
                    'depth': len(chain),
                    'path': [c.id for c in chain]
                })
                fixed_comment = RedditComment.from_dict(comment_dict)
                valid_comments.append(fixed_comment)
                comment_by_id[fixed_comment.id] = fixed_comment

        # Sort comments in depth-first order
        def get_sort_key(comment):
            """Return tuple for sorting in depth-first order"""
            return (
                tuple(comment.path),  # Primary sort by path
                comment.created_utc  # Secondary sort by creation time within same path
            )

        valid_comments.sort(key=get_sort_key)

        # Update max_depth based on actual maximum depth in valid comments
        actual_max_depth = max((comment.depth for comment in valid_comments), default=0)

        return Thread(self.post, valid_comments, actual_max_depth)

    def renew_max_depth(self):
        depths = [len(c.path) for c in self.comments]
        self.max_depth = max(1, max(depths))

    def to_dict(self) -> Dict:
        """Convert thread to dictionary format with all fields"""
        # Get post data iteratively from asdict
        post_data = asdict(self.post)
        post_dict = {f'{k}': v for k, v in post_data.items()}

        # Get comments data iteratively
        comments_list = []
        for comment in self.comments:
            comment_dict = comment.to_dict()
            comments_list.append(comment_dict)

        return {
            'post': post_dict,
            'comments': comments_list
        }

    def to_flat_dict(self) -> Dict:
        """Convert thread to flattened dictionary format"""
        # Get post data
        post_data = asdict(self.post)
        thread_dict = {f'post_{k}': v for k, v in post_data.items()}

        # Add comments data by depth
        for i, comment in enumerate(self.comments):
            comment_dict = comment.to_dict()
            for k, v in comment_dict.items():
                thread_dict[f'comment_{i}_{k}'] = v

        thread_dict['max_depth'] = self.max_depth
        return thread_dict

    def to_dataframe(self) -> pd.DataFrame:
        """Convert single thread to DataFrame"""
        return pd.DataFrame([self.to_flat_dict()])

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> List['Thread']:
        """Create Thread objects from DataFrame"""
        threads = []

        for _, row in df.iterrows():
            # Extract post fields
            post_dict = {k[5:]: v for k, v in row.items()
                         if k.startswith('post_')}
            post = RedditPost.from_dict(post_dict)

            # Extract comment fields
            comments = []
            comment_index = 0
            while True:
                comment_dict = {
                    k[len(f'comment_{comment_index}_'):]: v
                    for k, v in row.items()
                    if k.startswith(f'comment_{comment_index}_')
                }
                if comment_dict:
                    comments.append(RedditComment.from_dict(comment_dict))
                    comment_index = comment_index + 1
                else:
                    break

            threads.append(cls(post=post, comments=comments, max_depth=row['max_depth']))

        return threads

    def save_to_json(self, file_path: str):
        """Save thread to JSON file"""
        data = {
            'post': self.post.to_dict(),
            'comments': [c.to_dict() for c in self.comments],
            'max_depth': self.max_depth
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_from_json(cls, file_path: str) -> 'Thread':
        """Load thread from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return cls(
            post=RedditPost.from_dict(data['post']),
            comments=[RedditComment.from_dict(c) for c in data['comments']],
            max_depth=data['max_depth']
        )

    @classmethod
    def load_from_batch_json(cls, file_path: str) -> List['Thread']:
        """Load thread from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        threads_data = data['threads']
        result_list = []
        for thread_json in threads_data:
            thread = cls(
                post=RedditPost.from_dict(thread_json['post']),
                comments=[RedditComment.from_dict(c) for c in thread_json['comments']],
                max_depth=thread_json['max_depth']
            )
            result_list.append(thread)

        return result_list

    @classmethod
    def from_dict(cls, thread_dict: Dict) -> 'Thread':
        """Create Thread from dictionary"""
        thread = cls(
            post=RedditPost.from_dict(thread_dict['post']),
            comments=[RedditComment.from_dict(c) for c in thread_dict['comments']],
            max_depth=1
        )
        thread.make_complete()
        return thread

    def __str__(self):
        thread_dict = {
            "post": {
                # "author": self.post.author,
                "subreddit": self.post.subreddit,
                "title": self.post.title,
                "selftext": self.post.selftext,
                # "score": self.post.score
            },
            "comments": [
                {
                    # "author": comment.author,
                    "body": comment.body,
                    # "score": comment.score,
                    "id": comment.id,
                    "path": comment.path
                } for comment in self.comments
            ]
        }
        return json.dumps(thread_dict)


class RedditThreadDataset(Dataset): 
    def __init__(self, data_dir, task_name, cache_dir, **kwargs):

        # Load or create base threads
        threads_cache = Path(cache_dir) / 'reddit_threads_cache.pkl'
        lock_path = f"{threads_cache}.lock"
        
        if os.path.exists(threads_cache):
            print(f'Loading cached reddit threads from {threads_cache}')
            with open(threads_cache, 'rb') as f:
                self.threads = pickle.load(f)
        else:
            print(f'Creating cache file for reddit threads...')
            threads = []
            with FileLock(lock_path):
                data_fnames = sorted(glob(str(Path(data_dir) / task_name / 'batch_*.json')), 
                                   key=lambda x: int(re.search(r'batch_(\d+)', x).group(1)))

                for data_fname in tqdm(data_fnames):
                    with open(data_fname, 'r', encoding='utf-8') as f:
                        batch_data = json.load(f)

                    for thread_data in batch_data['threads']:
                        thread = Thread(
                            post=RedditPost.from_dict(thread_data['post']),
                            comments=[RedditComment.from_dict(c) for c in thread_data['comments']],
                            max_depth=thread_data['max_depth']
                        ) # .make_complete()
                        threads.append(thread)

            os.makedirs(os.path.dirname(threads_cache), exist_ok=True)
            with open(threads_cache, 'wb') as f:
                pickle.dump(threads, f)
            self.threads = threads

        # Load or create specialized views
        views_cache = Path(cache_dir) / 'reddit_thread_dict_cache.pkl'
        if os.path.exists(views_cache):
            print(f'Loading cached thread views from {views_cache}')
            with open(views_cache, 'rb') as f:
                cached_views = pickle.load(f)
        else:
            print(f'Creating cache file for thread views...')
            cached_views = {
                'sit': self._create_single_comments(),
                'set': self._create_single_threads(),
                'ft': self._create_full_threads()
            }
            with open(views_cache, 'wb') as f:
                pickle.dump(cached_views, f)

        cached_views['sit'] = sum([cached_views['sit'][i::2000] for i in range(10)], []) # sum([cached_views['sit'][i::10] for i in range(10)], [])
        cached_views['set'] = sum([cached_views['set'][i::500] for i in range(10)], [])
        cached_views['ft'] = sum([cached_views['set'][i::100] for i in range(10)], [])

        self.thread_dict = cached_views


    def _create_single_comments(self):
        """Create threads with only root comments"""
        single_comments = []
        for thread in self.threads:
            root_comments = [c for c in thread.comments if c.depth == 1]
            for root_comment in root_comments:
                single_comments.append(Thread(
                    post=thread.post,
                    comments=[root_comment],
                    max_depth=1
                ))
        return single_comments

    def _create_single_threads(self):
        """Create threads with single paths truncated to depth 5"""
        single_threads = []
        seen_root_comments = set()
        
        for thread in self.threads:
            # Group comments by root comment
            root_comment_groups = {}
            for comment in thread.comments:
                if comment.depth == 1:
                    if comment.id not in seen_root_comments:
                        root_comment_groups[comment.id] = [comment]
                        seen_root_comments.add(comment.id)
                elif comment.path[0] in root_comment_groups:
                    root_comment_groups[comment.path[0]].append(comment)

            # Process each root comment group
            for root_id, group in root_comment_groups.items():
                if len(group) <= 1:  # Skip if only root comment
                    continue
                    
                # Find all leaf nodes (comments that don't have children)
                comment_ids = {c.id for c in group}
                parent_ids = {c.parent_id for c in group if c.depth > 1}
                leaf_comments = [c for c in group if c.id not in parent_ids and c.depth > 1 and c.depth < 6]
                
               # For each leaf, create a complete path using the path attribute
                for leaf in leaf_comments:
                    # Get all comment IDs in the path from root to this leaf
                    path_ids = leaf.path
                    
                    # Find the actual comment objects for each ID in the path
                    comment_path = []
                    for i, comment_id in enumerate(path_ids):
                        comment = next((c for c in group if c.id == comment_id), None)
                        if comment:
                            comment_path.append(comment)
                    
                    if comment_path:
                        single_threads.append(Thread(
                            post=thread.post,
                            comments=comment_path,
                            max_depth=min(5, max(c.depth for c in comment_path))
                        ))
        
        return single_threads

    def _create_full_threads(self):
        """Create threads truncated to 5 root comments and depth 5"""
        full_threads = []
        
        for thread in self.threads:
            # First get all comments with their corresponding subtree depths
            comment_depths = {}
            max_depths = {}  # Maximum depth for each root comment's subtree
            
            # Calculate max depth of subtree for each comment
            for comment in thread.comments:
                comment_depths[comment.id] = comment.depth
                if comment.depth == 1:  # If root comment
                    max_depths[comment.id] = 1
                else:
                    root_id = comment.path[0]
                    if root_id in max_depths:
                        max_depths[root_id] = max(max_depths[root_id], comment.depth)

            # Get root comments sorted by their subtree depth (primary) and creation time (secondary)
            root_comments = sorted(
                [c for c in thread.comments if c.depth == 1],
                key=lambda x: (-max_depths[x.id], x.created_utc)
            )[:5]  # Take top 5 root comments with deepest subtrees
            
            if not root_comments:
                continue
                
            # Get all descendant comments up to depth 5
            selected_comments = []
            root_ids = {c.id for c in root_comments}
            
            # Add root comments first
            selected_comments.extend(root_comments)
            
            # Then add their descendants up to depth 5
            for comment in thread.comments:
                if (comment.depth <= 5 and 
                    comment.path[0] in root_ids and 
                    comment not in selected_comments):
                    selected_comments.append(comment)
            
            # Sort selected comments to maintain conversation flow
            selected_comments.sort(key=lambda x: (x.path, x.created_utc))
            
            full_threads.append(Thread(
                post=thread.post,
                comments=selected_comments,
                max_depth=min(5, max(c.depth for c in selected_comments))
            ))
        
        return full_threads

    def format_thread(self, thread: Thread) -> str:
        """Format thread as readable text with important fields"""
        output = [
            f"Subreddit: r/{thread.post.subreddit}",
            f"Author: {thread.post.author_fullname}",
            f"Title: {thread.post.title}",
            f"Post: {thread.post.selftext}\n"
        ]
        
        for comment in thread.comments:
            indent = "  " * (comment.depth - 1)
            output.append(f"{indent}Comment by {comment.author_fullname}: {comment.body}")
            
        return "\n".join(output)

    def __len__(self):
        return len(self.threads)

    def __getitem__(self, idx):
        return self.threads[idx]



class ELF22Dataset(Dataset):
    def __init__(self, data_dir, task_name, **kwargs):
        df_dict = {
            'train': pd.read_json(os.path.join(data_dir, task_name, 'train.json')),
            'validation': pd.read_json(os.path.join(data_dir, task_name, 'eval.json')),
            'test': pd.read_json(os.path.join(data_dir, task_name, 'test.json')),
        }

        # 1. convert flag name
        for stage, df in df_dict.items():
            df_dict[stage] = convert_deprecated_flags(df)

        # remove empty rows
        for stage, df in df_dict.items():
            df['Input'].replace('', np.nan, inplace=True)
            df['Output'].replace('', np.nan, inplace=True)
            df.dropna(subset=['Input', 'Output'], inplace=True)
            print(f"after dropping empty rows: {len(df)}")

            df['Context'] = 'Context: ' + 'r/' + df['Subreddit'] + ' ' + df['Title'] + ' ' + df['Post']
            df_dict[stage] = self.preprocess_input_and_output(df)

        for stage, df in df_dict.items():
            if not 'sample_index' in df:
                df_dict[stage]['sample_index'] = df_dict[stage].index
            df_dict[stage] = self.to_thread_format(df_dict[stage])

        self.datasets = DatasetDict({
            'train': Dataset.from_pandas(df_dict['train']),
            'validation': Dataset.from_pandas(df_dict['validation']),
            'test': Dataset.from_pandas(df_dict['test']),
        })

    def __len__(self, stage='train'):
        return len(self.datasets[stage])

    def __getitem__(self, idx, stage='train'):
        return self.datasets[stage].loc[idx]

    def preprocess_input_and_output(self, df):
        df['Output_preprocessed'] = df['Flag'] + ' ' + df['Output']
        df['InputContent'] = 'Context: r/' + df['Subreddit'] + '\nTitle: ' + df['Title'] + '\nPost: ' + df['Post'] + '\nComment: ' + df['Troll']
        return df

    def to_thread_format(self, df):
        def generate_reddit_id(length=6):
            """Generate a random alphanumeric ID similar to Reddit's base36 format"""
            return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

        input_threads = []
        chosen_threads = []
        chosen_comments = []
        post_id = generate_reddit_id(6)  # Post IDs are usually 6 chars
        comment_id = generate_reddit_id(7)  # Comment IDs are usually 7-8 chars
        subreddit_id = generate_reddit_id(5)  # The part after t5_
        chosen_comment_id = generate_reddit_id(7)
        post_created_utc = int(np.random.randint(1600000000, 1700000000))
        comment_created_utc = post_created_utc + int(np.random.randint(10, 3600))
        chosen_comment_created_utc = comment_created_utc + int(np.random.randint(10, 3600))
        reject_comment_created_utc = comment_created_utc + int(np.random.randint(10, 3600))
        retrieved_utc = max(chosen_comment_created_utc, reject_comment_created_utc) + int(np.random.randint(60, 36000))
        subreddit_subscribers = np.random.randint(1000, 1000000)

        for _, row in df.iterrows():
            post = RedditPost.from_dict({
                'author': 'postuser1',
                'author_fullname': 't2_postuser1',
                'author_created_utc': int(np.random.randint(1000000000, 1600000000)),
                'created_utc': post_created_utc,
                'retrieved_utc': retrieved_utc,
                'id': post_id,
                'subreddit': row['Subreddit'],
                'subreddit_id': f't5_{subreddit_id}',
                'title': row['Title'],
                'selftext': row['Post'],
                'permalink': f'/r/{row["Subreddit"]}/comments/{post_id}/{row["Title"].replace(" ", "_")}/',
                'score': 1,
                'subreddit_subscribers': subreddit_subscribers,
                'num_comments': 1,
                'url': f'https://www.reddit.com/r/{row["Subreddit"]}/comments/{post_id}/{row["Title"].replace(" ", "_")}/',
            })

            comment = RedditComment.from_dict({
                'author': 'commentuser1',
                'author_fullname': 't2_commentuser1',
                'author_created_utc': int(np.random.randint(1000000000, 1600000000)),
                'created_utc': comment_created_utc,
                'retrieved_on': retrieved_utc,
                'link_id': f't3_{post_id}',
                'parent_id': f't3_{post_id}',
                'id': comment_id,
                'subreddit': row['Subreddit'],
                'subreddit_id': f't5_{subreddit_id}',
                'body': row['Troll'],
                'permalink': f'/r/{row["Subreddit"]}/comments/{post_id}/{row["Title"].replace(" ", "_")}/{comment_id}/',
                'score': row.get('Score', 1) if pd.notna(row.get('Score', 1)) else 1,
                'depth': 1,
                'path': [comment_id],
                'url': f'https://www.reddit.com/r/{row["Subreddit"]}/comments/{post_id}/{row["Title"].replace(" ", "_")}/{comment_id}/',
            })

            chosen_comment = RedditComment.from_dict({
                'author': 'commentuser2',
                'author_fullname': 't2_commentuser2',
                'author_created_utc': int(np.random.randint(1000000000, 1600000000)),
                'created_utc': chosen_comment_created_utc,
                'retrieved_on': retrieved_utc,
                'link_id': f't3_{post_id}',
                'parent_id': f't1_{comment_id}',
                'id': chosen_comment_id,
                'subreddit': row['Subreddit'],
                'subreddit_id': f't5_{subreddit_id}',
                'body': row['Response'],
                'permalink': f'/r/{row["Subreddit"]}/comments/{post_id}/{row["Title"].replace(" ", "_")}/{chosen_comment_id}/',
                'score': row.get('Score', 1) if pd.notna(row.get('Score', 1)) else 1,
                'depth': 2,
                'path': [comment_id, chosen_comment_id],
                'url': f'https://www.reddit.com/r/{row["Subreddit"]}/comments/{post_id}/{row["Title"].replace(" ", "_")}/{chosen_comment_id}/',
            })

            input_thread = str(Thread(post=post, comments=[comment], max_depth=1))
            chosen_comment_tmp = json.loads(str(chosen_comment))
            # chosen_comment_tmp['TrollingStrategy'] = ""           # M0
            chosen_comment_tmp['ResponseStrategy'] = row['Flag']     # M1
            # chosen_comment_tmp['Reasoning'] = row['TrollAnalysis'] if row['FlagTroll'] == '[nontroll]' else row['ResponseAnalysis']
            chosen_comment_tmp = json.dumps(chosen_comment_tmp)
            chosen_thread = json.loads(input_thread)
            chosen_thread['comments'].append(json.loads(chosen_comment_tmp))
            chosen_thread = json.dumps(chosen_thread)
            
            input_threads.append(input_thread)
            chosen_comments.append(chosen_comment_tmp)
            chosen_threads.append(chosen_thread)

        df['InputThread'] = input_threads
        df['ChosenComment'] = chosen_comments
        df['ChosenThread'] = chosen_threads

        return df


class ELFHPDataset(Dataset):
    def __init__(self, **kwargs):
        dataset = load_dataset("huijelee/ELF-HP")

        processed_splits = {}
        for split_name in ['train', 'test']:
            df = dataset[split_name].to_pandas()
            df_threaded = self.to_thread_format(df)
            
            df_threaded['InputContent'] = 'Context: r/' + df_threaded['Subreddit'] + '\nTitle: ' + df_threaded['Title'] + '\nPost: ' + df_threaded['Post'] + '\nComment: ' + df_threaded['Troll']
            df_threaded['output'] = df_threaded['ChosenComment']
            df_threaded['Context'] = 'Context: r/' + df_threaded['Subreddit'] + '\nTitle: ' + df_threaded['Title'] + '\nPost: ' + df_threaded['Post']
            
            processed_splits[split_name] = Dataset.from_pandas(df_threaded)

        self.datasets = DatasetDict({
            'train': processed_splits['train'],
            'validation': processed_splits['test'],
            'test': processed_splits['test'],
        })

    def __len__(self, stage='train'):
        return len(self.datasets[stage])

    def __getitem__(self, idx, stage='train'):
        return self.datasets[stage][idx]

    def to_thread_format(self, df):
        def generate_reddit_id(length=6):
            """Generate a random alphanumeric ID similar to Reddit's base36 format"""
            return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

        input_threads = []
        chosen_threads = []
        chosen_comments = []
        post_id = generate_reddit_id(6)  # Post IDs are usually 6 chars
        comment_id = generate_reddit_id(7)  # Comment IDs are usually 7-8 chars
        subreddit_id = generate_reddit_id(5)  # The part after t5_
        chosen_comment_id = generate_reddit_id(7)
        post_created_utc = int(np.random.randint(1600000000, 1700000000))
        comment_created_utc = post_created_utc + int(np.random.randint(10, 3600))
        chosen_comment_created_utc = comment_created_utc + int(np.random.randint(10, 3600))
        reject_comment_created_utc = comment_created_utc + int(np.random.randint(10, 3600))
        retrieved_utc = max(chosen_comment_created_utc, reject_comment_created_utc) + int(np.random.randint(60, 36000))
        subreddit_subscribers = np.random.randint(1000, 1000000)

        for _, row in df.iterrows():
            post = RedditPost.from_dict({
                'author': 'postuser1',
                'author_fullname': 't2_postuser1',
                'author_created_utc': int(np.random.randint(1000000000, 1600000000)),
                'created_utc': post_created_utc,
                'retrieved_utc': retrieved_utc,
                'id': post_id,
                'subreddit': row['Subreddit'],
                'subreddit_id': f't5_{subreddit_id}',
                'title': row['Title'],
                'selftext': row['Post'],
                'permalink': f'/r/{row["Subreddit"]}/comments/{post_id}/{row["Title"].replace(" ", "_")}/',
                'score': 1,
                'subreddit_subscribers': subreddit_subscribers,
                'num_comments': 1,
                'url': f'https://www.reddit.com/r/{row["Subreddit"]}/comments/{post_id}/{row["Title"].replace(" ", "_")}/',
            })

            comment = RedditComment.from_dict({
                'author': 'commentuser1',
                'author_fullname': 't2_commentuser1',
                'author_created_utc': int(np.random.randint(1000000000, 1600000000)),
                'created_utc': comment_created_utc,
                'retrieved_on': retrieved_utc,
                'link_id': f't3_{post_id}',
                'parent_id': f't3_{post_id}',
                'id': comment_id,
                'subreddit': row['Subreddit'],
                'subreddit_id': f't5_{subreddit_id}',
                'body': row['Troll'],
                'permalink': f'/r/{row["Subreddit"]}/comments/{post_id}/{row["Title"].replace(" ", "_")}/{comment_id}/',
                'score': row['Score'] if pd.notna(row['Score']) else 1,
                'depth': 1,
                'path': [comment_id],
                'url': f'https://www.reddit.com/r/{row["Subreddit"]}/comments/{post_id}/{row["Title"].replace(" ", "_")}/{comment_id}/',
            })

            chosen_comment = RedditComment.from_dict({
                'author': 'commentuser2',
                'author_fullname': 't2_commentuser2',
                'author_created_utc': int(np.random.randint(1000000000, 1600000000)),
                'created_utc': chosen_comment_created_utc,
                'retrieved_on': retrieved_utc,
                'link_id': f't3_{post_id}',
                'parent_id': f't1_{comment_id}',
                'id': chosen_comment_id,
                'subreddit': row['Subreddit'],
                'subreddit_id': f't5_{subreddit_id}',
                'body': row['ChosenResponse'],
                'permalink': f'/r/{row["Subreddit"]}/comments/{post_id}/{row["Title"].replace(" ", "_")}/{chosen_comment_id}/',
                'score': row.get('Score', 1) if pd.notna(row.get('Score', 1)) else 1,
                'depth': 2,
                'path': [comment_id, chosen_comment_id],
                'url': f'https://www.reddit.com/r/{row["Subreddit"]}/comments/{post_id}/{row["Title"].replace(" ", "_")}/{chosen_comment_id}/',
            })

            input_thread = str(Thread(post=post, comments=[comment], max_depth=1))
            chosen_comment_tmp = json.loads(str(chosen_comment))
            chosen_comment_tmp['TrollingStrategy'] = row['FlagTS']           
            chosen_comment_tmp['ResponseStrategy'] = row['ChosenFlagRS']     
            chosen_comment_tmp = json.dumps(chosen_comment_tmp)
            chosen_thread = json.loads(input_thread)
            chosen_thread['comments'].append(json.loads(chosen_comment_tmp))
            chosen_thread = json.dumps(chosen_thread)
            
            input_threads.append(input_thread)
            chosen_comments.append(chosen_comment_tmp)
            chosen_threads.append(chosen_thread)

        df['InputThread'] = input_threads
        df['ChosenComment'] = chosen_comments # [-1] or str({}) or None
        df['ChosenThread'] = chosen_threads

        return df


class QianDataset:
    def __init__(self, file_path, name):
        self.name = name
        self.file_path = file_path

    def get_dataframe(self) -> pd.DataFrame:
        print(f"Loading '{self.name}' dataset from {self.file_path}...")
        self.df = pd.read_csv(self.file_path)
        
        all_hateful_turns_with_context = []
        for _, row in self.df.iterrows():
            text = row['text']
            hate_indices_str = row['hate_speech_idx']
            
            turns = re.split(r'\n\d+\.\s', '\n' + text)
            turns = [turn.strip() for turn in turns if turn.strip()]
            
            troll_indices = []
            if pd.notna(hate_indices_str) and hate_indices_str != 'n/a':
                troll_indices = [int(i) for i in re.findall(r'\d+', str(hate_indices_str))]

            for troll_idx in troll_indices:
                if 1 <= troll_idx <= len(turns):
                    troll_text = turns[troll_idx - 1]
                    
                    previous_turns = turns[:troll_idx - 1]
                    formatted_turns = [f"{turn}" for turn in previous_turns]
                    context = "\n".join(formatted_turns)

                    text = 'Context: ' + context + '\nComment: ' + troll_text
                    
                    all_hateful_turns_with_context.append({
                        'text': text, 
                        'Context': context,
                        'troll_text': troll_text,
                        'label': 1
                    })

        print(f"{len(all_hateful_turns_with_context)} hateful turns with context are loaded from {self.name}")
        return pd.DataFrame(all_hateful_turns_with_context)



class CADDDataset(Dataset):
    def __init__(self, data_dir, task_name, preprocess=True):
        self.data_dir = data_dir
        self.task_name = task_name
        
        # Load CADD JSON files
        try:
            self.datasets = {
                'train': pd.read_json(os.path.join(data_dir, task_name, 'CADD_train.json'), orient='index'),
                'validation': pd.read_json(os.path.join(data_dir, task_name, 'CADD_dev.json'), orient='index'),
                'test': pd.read_json(os.path.join(data_dir, task_name, 'CADD_test.json'), orient='index'),
            }
        except ValueError:
            # Fallback for standard list json
            self.datasets = {
                'train': pd.read_json(os.path.join(data_dir, task_name, 'CADD_train.json')),
                'validation': pd.read_json(os.path.join(data_dir, task_name, 'CADD_dev.json')),
                'test': pd.read_json(os.path.join(data_dir, task_name, 'CADD_test.json')),
            }

        for stage in ['train', 'validation', 'test']:
            df = self.datasets[stage]
            
            # Normalize keys to Standard: Title, Post, Troll
            df.rename(columns={
                'title': 'Title', 
                'body': 'Post', 
                'comment': 'Troll'
            }, inplace=True)

            if 'Subreddit' not in df.columns:
                df['Subreddit'] = 'unknown' # Default for CADD
            if 'Response' not in df.columns:
                df['Response'] = "" # Default empty response

            # Fill NaNs
            df['Title'].fillna('', inplace=True)
            df['Post'].fillna('', inplace=True)
            df['Troll'].fillna('', inplace=True)

            # Drop Invalid Rows (Empty Inputs or Output)
            df = df[((df['Post'] != '') | (df['Title'] != '')) & (df['Troll'] != '')]

            # Create Context String
            df['Context'] = 'Context: r/' + df['Subreddit'] + ' ' + df['Title'] + ' ' + df['Post']

            # Generate Fake Thread Structure
            if preprocess:
                df = self.to_thread_format(df)

            self.datasets[stage] = df

    def __len__(self, stage='train'):
        return len(self.datasets[stage])

    def __getitem__(self, idx, stage='train'):
        if isinstance(idx, int):
            return self.datasets[stage].iloc[idx].to_dict()
        return self.datasets[stage].loc[idx].to_dict()

    def to_thread_format(self, df):
        def generate_reddit_id(length=6):
            return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

        input_threads = []
        chosen_threads = []
        chosen_comments = []
        
        for _, row in df.iterrows():
            post_id = generate_reddit_id(6)
            comment_id = generate_reddit_id(7)
            chosen_comment_id = generate_reddit_id(7)
            subreddit_id = generate_reddit_id(5)
            
            # Random Timestamps
            post_created_utc = int(np.random.randint(1600000000, 1700000000))
            comment_created_utc = post_created_utc + int(np.random.randint(10, 3600))
            chosen_comment_created_utc = comment_created_utc + int(np.random.randint(10, 3600))
            retrieved_utc = chosen_comment_created_utc + int(np.random.randint(60, 36000))
            subreddit_subscribers = np.random.randint(1000, 1000000)

            # Fake Subreddit URL/Permalink
            sub_name = row['Subreddit'] if row['Subreddit'] else "unknown"
            
            # 1. Post
            post = RedditPost.from_dict({
                'author': 'postuser1',
                'author_fullname': 't2_postuser1',
                'author_created_utc': int(np.random.randint(1000000000, 1600000000)),
                'created_utc': post_created_utc,
                'retrieved_utc': retrieved_utc,
                'id': post_id,
                'subreddit': sub_name,
                'subreddit_id': f't5_{subreddit_id}',
                'title': row['Title'],
                'selftext': row['Post'],
                'permalink': f'/r/{sub_name}/comments/{post_id}/',
                'score': 1,
                'subreddit_subscribers': subreddit_subscribers,
                'num_comments': 1,
                'url': f'https://www.reddit.com/r/{sub_name}/comments/{post_id}/',
            })

            # 2. Harmful Comment (Troll)
            comment = RedditComment.from_dict({
                'author': 'harmfuluser1',
                'author_fullname': 't2_harmfuluser1',
                'author_created_utc': int(np.random.randint(1000000000, 1600000000)),
                'created_utc': comment_created_utc,
                'retrieved_on': retrieved_utc,
                'link_id': f't3_{post_id}',
                'parent_id': f't3_{post_id}',
                'id': comment_id,
                'subreddit': sub_name,
                'subreddit_id': f't5_{subreddit_id}',
                'body': row['Troll'],
                'permalink': f'/r/{sub_name}/comments/{post_id}/_/{comment_id}/',
                'score': 1,
                'depth': 1,
                'path': [comment_id],
                'url': f'https://www.reddit.com/r/{sub_name}/comments/{post_id}/_/{comment_id}/',
            })

            # 3. Response (Optional)
            response_body = row.get('Response', '')
            chosen_comment_dict = {}
            if response_body:
                chosen_comment = RedditComment.from_dict({
                    'author': 'counteruser2',
                    'author_fullname': 't2_counteruser2',
                    'author_created_utc': int(np.random.randint(1000000000, 1600000000)),
                    'created_utc': chosen_comment_created_utc,
                    'retrieved_on': retrieved_utc,
                    'link_id': f't3_{post_id}',
                    'parent_id': f't1_{comment_id}',
                    'id': chosen_comment_id,
                    'subreddit': sub_name,
                    'subreddit_id': f't5_{subreddit_id}',
                    'body': response_body,
                    'permalink': f'/r/{sub_name}/comments/{post_id}/_/{chosen_comment_id}/',
                    'score': 1,
                    'depth': 2,
                    'path': [comment_id, chosen_comment_id],
                    'url': f'https://www.reddit.com/r/{sub_name}/comments/{post_id}/_/{chosen_comment_id}/',
                })
                chosen_comment_dict = json.loads(str(chosen_comment))
            else:
                 chosen_comment_dict = {"body": "", "author": "[deleted]"}

            # Serialize
            input_thread = str(Thread(post=post, comments=[comment], max_depth=1))
            chosen_comment_str = json.dumps(chosen_comment_dict)
            
            chosen_thread_dict = json.loads(input_thread)
            if response_body:
                chosen_thread_dict['comments'].append(chosen_comment_dict)
            chosen_thread_str = json.dumps(chosen_thread_dict)

            input_threads.append(input_thread)
            chosen_comments.append(chosen_comment_str)
            chosen_threads.append(chosen_thread_str)

        df['InputThread'] = input_threads
        df['ChosenComment'] = chosen_comments
        df['ChosenThread'] = chosen_threads

        return df


dataset_dict = {
    'counter_trollingy': ELF22Dataset,
    'cadd': CADDDataset,
    'reddit_random_threads': RedditThreadDataset,
    'reddit_threads': RedditThreadDataset,
    'ELF-HP': ELFHPDataset,
    'qian_gab': QianDataset,
    'qian_reddit': QianDataset,
}


def main(hparams):
    datasets = dataset_dict[hparams.task_name](
        hparams.data_dir,
        hparams.task_name,
        preprocess=True,
        all_flags=hparams.generate_all_flags,
        with_flag=hparams.with_flag,
    ).datasets

    hparams.task_name = 'counter_trollingy'
    datasets = dataset_dict[hparams.task_name](
        hparams.data_dir,
        hparams.task_name,
        preprocess=True,
        all_flags=hparams.generate_all_flags,
        with_flag=hparams.with_flag,
    ).datasets
    pass


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(add_help=False)

    parser.add_argument("--data_dir", default="", type=str)
    parser.add_argument('--task_name', default='') # 'counter_trollingy')
    parser.add_argument('--generate_all_flags', action='store_true', default=False)
    parser.add_argument('--with_flag', action='store_true', default=False)

    hparams = parser.parse_args()

    main(hparams)
