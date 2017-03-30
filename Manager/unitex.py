import os
import marisa_trie
#from configuration import *
from configuration import extras

class Unitex():
    """
        Class contains methods for dealing with unitex delaf dictionary
    """

    def __init__(self):

        self.trie = marisa_trie.Trie()
        self.trie.load(extras['MarisaTree'])

    def lemma(self, token, pos='N'):
        if self.trie.has_keys_with_prefix(token + '$'):
            if pos and self.trie.has_keys_with_prefix(token + '$' + pos + '$'):
                return self.trie.keys(token + '$' + pos + '$')[0].split('$')[-2]
            elif pos and self.trie.has_keys_with_prefix(token + '$' + pos + '+'):
                return self.trie.keys(token + '$' + pos + '+')[0].split('$')[-2]
            else:
                return self.trie.keys(token + '$')[0].split('$')[-2]
        else:
            return token

    def morf(self, token, pos=None):
        if self.trie.has_keys_with_prefix(token + '$'):
            if pos and self.trie.has_keys_with_prefix(token + '$' + pos + '$'):
                return [v.split('$')[-1] for v in self.trie.keys(token + '$' + pos + '$')]
            elif pos and self.trie.has_keys_with_prefix(token + '$' + pos + '+'):
                return [v.split('$')[-1] for v in self.trie.keys(token + '$' + pos + '$')]
            else:
                return [v.split('$')[-1] for v in self.trie.keys(token + '$')]
        else:
            return ''
