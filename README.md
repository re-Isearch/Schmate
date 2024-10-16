# <IMAGE SRC="https://github.com/user-attachments/assets/54a16857-ed56-459c-94f4-abfd9f8ff8a1" HEIGHT=35> Schmate

Project שמאטע: Schmate* (Pronounced “SHMAH-teh)

Extending re-Isearch with vector datatypes for embeddings. Enabling a better dense passage retrieval (DPR) for retrieval augmented generation (RAG) and hybrid search.


## Abstract / Blurb

Schmate is the development name for the evolving next iteration of re-Isearch adding vector datatypes for embeddings and applications like retrieval augmented generation (RAG).
In contrast to typical vector stores the proposed re-Isearch+ shall offer a full passage information retrieval system (index and retrieval) using a combination of dense and sparse vectors as well as structure. It is dense passage retrieval (DPR) and a whole lot more. It addresses the stumbling blocks of chunking, has a tight integration of ingest, tokenisation, a number of alternative vector stores and similarity algorithms and, above all, uses a novel combination of understanding document structure (implicit and explicit) to provide a better contextual passage retrieval to solve the problem of misaligned context. This builds on the observation that meaning is also communicated through structure so needs to be viewed in the context of structure. Since structure like the words are meant by the sender (writer) to be received and understood (reader) our approach is to exploit the original author's organization of content to determine appropriate passages rather than relying solely on the chunks.


## Project

Re-Isearch is a 100% open source (Apache 2.0) novel multimodal search and retrieval engine using mathematical models and algorithms different from the all-too-common inverted index. It is a kind of hybrid between full-text, XML, object and graph noSQL-db that natively ingests a wide range of document types and formats. It has been open-sourced through a grant from Nlnet/NGI-Zero Search. See our talk from FOSDEM ‘22: A lightning intro to re-Isearch.
https://archive.fosdem.org/2022/schedule/event/lt_re_lsearch/

The re-Isearch engine exploits document structure, both implicit (XML and other markup) and explicit (visual groupings such as paragraph), to zero in on relevant sections of documents, not just links to documents. These are a heterogeneous mix of text, data (a large number of datatypes including: numerical, computed, range, date, time, geo, boolean etc. as well as a number of hashes including several phonetic), network objects and databases. These datatypes have their own, for their individual datatypes, storage and retrieval algorithms (including relevant ranking and similarity methods). Project Schmate intends to extend re-Isearch with vector datatypes tuned for embeddings.

These new datatypes are intended to provide a powerful alternative to popular vector databases like FAISS for Dense Passage Retrieval (DPR) in, among other domains, Retrieval Augmented Generation (RAG) applied to popular open source context constrained large language models such as LLaMa and Mistral. Typical vector databases used for DPR (Dense Passage Retrieval) demand a prior segmentation of content into size constrained blobs or passages. This often results in context mismatch and disappointment.

This is where re-Isearch and its proposed new datatypes and extensions enter. Since re-Isearch has a fully dynamic unit of retrieval, definable at search time or by heuristic, its simplifies the creation and maintenance of DPR systems and provides a significant advantage for, among other applications, RAG. There are, of course, a myriad of other uses.

Instead of purely using the retrieved passages and feeding them into the LLM we look at the scope of the retrieved passages to determine a more optimal passage response. Our approach is to exploit data (document) structure to define the chunks. On the search retrieval side (full-text retrieval) alongside the search for similar chunks we have a retrieval of its own model of relevant bits using its ability to fully retrieve structural elements without need to re-parse: dynamic unit of retrieval. This approach maintains the original author's organization of content and helps keep the text coherent. It builds on the idea that documents are implicitly structured to be understood by humans using either explicit markup or implicit structure such as lines, sentences or paragraphs. It also builds on the notion that meaning is communicated through also structure so needs to be viewed in the context of structure. Rather than just use the chunk we can use the contextual scope—with, of course, a size constraint definable at search time—and also, when desired or warranted provide the traceback.

Since the re-Isearch  engine can at search-time respond to geo-location, user rights or other issues which may define what constitutes as “inappropriate” as it is just toggling a single bit. This feature can be quite useful for retrieval augmentation.





(*) Schmate (Pronounced “SHMAH-teh) is Yiddish for rag (שמאטע).



<BR>
<BR>

![image](https://github.com/user-attachments/assets/4a8ad719-8338-42c9-9d67-11505a939a0e)

<IMG SRC="https://nlnet.nl/image/logo_nlnet.svg" ALT="NLnet Foundation" height=100> &nbsp; &nbsp; <IMG SRC="https://ngi.eu/wp-content/uploads/sites/77/2017/10/bandiera_stelle.png" ALT="EU" height=100> <IMG SRC="https://github.com/user-attachments/assets/f08b708d-35fe-4331-8eff-715fa3fbfe84" ALT="CH" height=100>



<BR>
This project was funded through the NGI0 Commons Fund, a fund established by NLnet with financial support from the European Commission's Next Generation Internet programme, under the aegis of DG Communications Networks, Content and Technology under grant agreement No 101135429. Additional funding is made available by the Swiss State Secretariat for Education, Research and Innovation (SERI).
