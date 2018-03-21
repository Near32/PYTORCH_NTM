from __future__ import print_function
from PIL import Image
from os.path import join
import os
import random
import torch.utils.data as data
from utils import download_url, check_integrity, list_dir, list_files
import numpy as np

def onehotencoded(y, nbrClass) :
    r = np.zeros(nbrClass)
    r[y] = 1
    return r 

class Omniglot(data.Dataset):
    """`Omniglot <https://github.com/brendenlake/omniglot>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``omniglot-py`` exists.
        background (bool, optional): If True, creates dataset from the "background" set, otherwise
            creates from the "evaluation" set. This terminology is defined by the authors.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset zip files from the internet and
            puts it in root directory. If the zip files are already downloaded, they are not
            downloaded again.
    """
    folder = 'omniglot-py'
    download_url_prefix = 'https://github.com/brendenlake/omniglot/raw/master/python'
    zips_md5 = {
        'images_background': '68d2efa1b9178cc56df9314c21c6e718',
        'images_evaluation': '6b91aef0f799c5bb55b94e3f2daec811'
    }

    def __init__(self, root, background=True,
                 transform=None, target_transform=None,
                 download=False):
        self.root = join(os.path.expanduser(root), self.folder)
        self.background = background
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.target_folder = join(self.root, self._get_target_folder())
        self._alphabets = list_dir(self.target_folder)
        self.alphabet2char = dict()
        self.alphabet2listChar = dict()
        
        print('Alphabets :')
        for idxa, a in enumerate(self._alphabets) :
            print(idxa,a)
            self.alphabet2char[a] = [ join(a, c) for c in list_dir(join(self.target_folder, a))]
            self.alphabet2listChar[a] = dict()
            for idxc, path in enumerate(self.alphabet2char[a]) :
                listChar = [ (join(self.target_folder,path,image), idxc, idxa)   for image in list_files(join(self.target_folder, path), '.png')]
                self.alphabet2listChar[a][path] = listChar
                #print(listChar)

        self._characters = sum([[join(a, c) for c in list_dir(join(self.target_folder, a))]
                                for a in self._alphabets], [])
        self._character_images = [[(image, idx) for image in list_files(join(self.target_folder, character), '.png')]
                                  for idx, character in enumerate(self._characters)]
        self._flat_character_images = sum(self._character_images, [])

    def __len__(self):
        return len(self._flat_character_images)

    def nbrAlphabets(self) :
        return len(self._alphabets)

    def sampleAlphabet(self, alphabet_idx) :
        alphabet_idx = alphabet_idx % self.nbrAlphabets()
        return self._alphabets[alphabet_idx]

    def nbrCharacters4Alphabet(self,alphabet_idx) :
        return len(self.alphabet2char[ self.sampleAlphabet(alphabet_idx) ] )

    def sampleCharacter4Alphabet(self, alphabet_idx, character_idx) :
        character_idx = character_idx % self.nbrCharacters4Alphabet(alphabet_idx)
        return self.alphabet2char[ self.sampleAlphabet(alphabet_idx) ][character_idx] 

    def nbrSamples4Character4Alphabet(self, alphabet_idx, character_idx) :
        a = self.sampleAlphabet(alphabet_idx)
        c = self.sampleCharacter4Alphabet(alphabet_idx,character_idx)
        return len(self.alphabet2listChar[a][c] )

    def sampleSample4Character4Alphabet(self, alphabet_idx, character_idx, sample_idx) :
        a = self.sampleAlphabet(alphabet_idx)
        c = self.sampleCharacter4Alphabet(alphabet_idx,character_idx)
        sample_idx = sample_idx % self.nbrSamples4Character4Alphabet(alphabet_idx=alphabet_idx,character_idx=character_idx)
        return self.alphabet2listChar[a][c][sample_idx]

    def generateFewShotLearningTask(self, alphabet_idx) :
        nbrChar = self.nbrCharacters4Alphabet(alphabet_idx)
        samples = list()
        for c in range(nbrChar) :
            nbrSample4c = self.nbrSamples4Character4Alphabet(alphabet_idx=alphabet_idx,character_idx=c)
            samples +=   [ {'alphabet':alphabet_idx, 'character':c, 'sample':idxsample, 'nbrCharacter':nbrChar} for idxsample in range(nbrSample4c) ]

        random.shuffle(samples)
        nbrSamples = len(samples)

        return samples, nbrChar, nbrSamples

    def generateIterFewShotLearningTask(self, alphabet_idx) :
        nbrChar = self.nbrCharacters4Alphabet(alphabet_idx)
        nbrSample4clist = list()
        samplesIt = list()
        for c in range(nbrChar) :
            nbrSample4c = self.nbrSamples4Character4Alphabet(alphabet_idx=alphabet_idx,character_idx=c)
            nbrSample4clist.append(nbrSample4c)
            samplesIt.append(c)

        minNbrSample4c = min(nbrSample4clist)
        random.shuffle(samplesIt)

        samples = list()
        for it in range(minNbrSample4c) :
            random.shuffle(samplesIt)
            for idxc in samplesIt :
                samples.append( {'alphabet':alphabet_idx, 'character':idxc, 'sample':it, 'nbrCharacter':nbrChar} )

        nbrSamples = len(samples)

        return samples, nbrChar, nbrSamples

    def generateIterFewShotInputSequence(self, alphabet_idx) :
        '''
        Returns :
            sequence of tuple (x_0, y_{-1}(dummy)), (x_1, y_0) ... (x_n, y_n-1), (x_n+1(dummy), y_n)
            nbr of characters in this task.
            nbr of samples in the whole task.
        '''
        samples, nbrChar, nbrSamples = self.generateIterFewShotLearningTask(alphabet_idx=alphabet_idx)
        seq = list()
        for i in range(nbrSamples) :
            d = dict()
            if i== 0 :
                x = samples[0]
                y = -1
            else :
                x = samples[i]
                y = samples[i-1]['character']
            dy = dict()
            dy['target'] = onehotencoded(y, nbrClass=nbrChar)
            
            d.update(x)
            d.update(dy)
            seq.append( d )
        
        # last sample's target regularization :
        dy = dict()
        dy['target'] = onehotencoded(samples[nbrSamples-1]['character'], nbrClass=nbrChar)
        d = dict()
        d.update(samples[0])
        d.update(dy)
        seq.append( d )

        return seq, nbrChar, nbrSamples+1  


    def getSample(self, alphabet_idx, character_idx, sample_idx) :
        image_path, character_class, idxa = self.sampleSample4Character4Alphabet( alphabet_idx, character_idx, sample_idx)
        image = Image.open(image_path, mode='r').convert('L')

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            character_class = self.target_transform(character_class)

        return image, character_class


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        image_name, character_class = self._flat_character_images[index]
        image_path = join(self.target_folder, self._characters[character_class], image_name)
        image = Image.open(image_path, mode='r').convert('L')

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            character_class = self.target_transform(character_class)

        return image, character_class

    def _check_integrity(self):
        zip_filename = self._get_target_folder()
        if not check_integrity(join(self.root, zip_filename + '.zip'), self.zips_md5[zip_filename]):
            return False
        return True

    def download(self):
        import zipfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        filename = self._get_target_folder()
        zip_filename = filename + '.zip'
        url = self.download_url_prefix + '/' + zip_filename
        download_url(url, self.root, zip_filename, self.zips_md5[filename])
        print('Extracting downloaded file: ' + join(self.root, zip_filename))
        with zipfile.ZipFile(join(self.root, zip_filename), 'r') as zip_file:
            zip_file.extractall(self.root)

    def _get_target_folder(self):
        return 'images_background' if self.background else 'images_evaluation'
