#!/usr/bin/env python
# -*- coding: utf-8 -*-
from unitex import Unitex
unitex = Unitex()


def lemma(token, pos='N'):
	token = u"%s" % token
	return unitex.lemma(token, pos)


def morf(token, pos=None):	
	return list(set(unitex.morf(token, pos)))


if __name__ == '__main__':
	
	#text = "meninas elas ála"
	#utext = u"%s" % text

	#print utext

	print (lemma(u'meninas'))
	print lemma(u"criancas")
