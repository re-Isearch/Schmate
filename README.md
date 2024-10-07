# Schmate

Project שמאטע: Schmate* (Pronounced “SHMAH-teh)

Extending re-Isearch with vector datatypes for embeddings. Enabling a better dense passage retrieval (DPR) for retrieval augmented generation (RAG) and hybrid search.

Re-Isearch is a 100% open source (Apache 2.0) novel multimodal search and retrieval engine using mathematical models and algorithms different from the all-too-common inverted index. It is a kind of hybrid between full-text, XML, object and graph noSQL-db that natively ingests a wide range of document types and formats. It has been open-sourced through a grant from Nlnet/NGI-Zero Search. See our talk from FOSDEM ‘22: A lightning intro to re-Isearch.
https://archive.fosdem.org/2022/schedule/event/lt_re_lsearch/

The re-Isearch engine exploits document structure, both implicit (XML and other markup) and explicit (visual groupings such as paragraph), to zero in on relevant sections of documents, not just links to documents. These are a heterogeneous mix of text, data (a large number of datatypes including: numerical, computed, range, date, time, geo, boolean etc. as well as a number of hashes including several phonetic), network objects and databases. These datatypes have their own, for their individual datatypes, storage and retrieval algorithms (including relevant ranking and similarity methods). Project Schmate intends to extend re-Isearch with a flat vector datatype tuned for embeddings.

This new datatypes are intended to provide a powerful alternative to popular vector databases like FAISS for Dense Passage Retrieval (DPR) in, among other domains, Retrieval Augmented Generation (RAG) applied to popular open source context constrained large language models.  LLaMa/LLaMa2/LLaMa3 have, for example, a relatively small 2k, resp. generally 4k and 8k context while  Mistral-7B has a context length of 32k, This is still too small to be able to include the whole local or updated corpus but only some bits (passages). RAG (Retrieval Augmented Generation) is a means to try to maneuver out of this constraint but because of the fixed unit of retrieval in typical vector databases used for DPR (Dense Passage Retrieval) they demand a prior segmentation of content into size constrained blobs or passages.

This is where re-Isearch and its proposed new datatypes and extensions enter. Since re-Isearch has a fully dynamic unit of retrieval, definable at search time or by heuristic, its simplifies the creation and maintenance of DPR systems and provides a significant advantage for, among other applications, RAG. There are, of course, a myriad of other uses.




(*) Schmate means RAG in Yiddish 
