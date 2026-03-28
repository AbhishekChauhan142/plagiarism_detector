"""
PAN Dataset Manager for plagiarism detection research.

Handles loading, parsing, and creating PAN-style corpora with XML annotations.
Official PAN datasets are available at https://pan.webis.de/
"""

import logging
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class PANDatasetManager:
    """Manages PAN plagiarism detection corpora.

    Supports parsing PAN XML annotation files, loading document pairs,
    computing corpus statistics, and creating sample corpora for testing.

    See https://pan.webis.de/ for official PAN shared task datasets.
    """

    # Sample texts for the synthetic corpus
    _SAMPLE_PAIRS = [
        {
            "type": "direct_copy",
            "source": (
                "Machine learning is a subset of artificial intelligence that provides systems "
                "the ability to automatically learn and improve from experience without being "
                "explicitly programmed. Machine learning focuses on the development of computer "
                "programs that can access data and use it to learn for themselves."
            ),
            "suspicious": (
                "Machine learning is a subset of artificial intelligence that provides systems "
                "the ability to automatically learn and improve from experience without being "
                "explicitly programmed. Machine learning focuses on the development of computer "
                "programs that can access data and use it to learn for themselves."
            ),
            "is_plagiarized": True,
        },
        {
            "type": "paraphrase",
            "source": (
                "Deep learning is part of a broader family of machine learning methods based on "
                "artificial neural networks with representation learning. Learning can be "
                "supervised, semi-supervised or unsupervised. Deep learning architectures such "
                "as deep neural networks and recurrent neural networks have been applied to "
                "fields including computer vision and natural language processing."
            ),
            "suspicious": (
                "Deep learning belongs to the wider group of machine learning techniques founded "
                "on artificial neural networks featuring representation learning. The learning "
                "process may be supervised, semi-supervised, or unsupervised. Deep neural "
                "networks and recurrent neural networks are examples of deep learning structures "
                "used in areas like image recognition and language understanding."
            ),
            "is_plagiarized": True,
        },
        {
            "type": "mosaic",
            "source": (
                "Natural language processing is a subfield of linguistics, computer science, "
                "and artificial intelligence concerned with the interactions between computers "
                "and human language. The goal is to program computers to process and analyze "
                "large amounts of natural language data."
            ),
            "suspicious": (
                "NLP is a subfield of computer science and AI that studies the interactions "
                "between computers and human language. Researchers aim to program computers "
                "to process large amounts of natural language data effectively, combining "
                "insights from linguistics and machine learning."
            ),
            "is_plagiarized": True,
        },
        {
            "type": "clean",
            "source": (
                "The Python programming language was created by Guido van Rossum and first "
                "released in 1991. Python is known for its simple syntax and readability, "
                "making it an excellent choice for beginners and experienced programmers alike."
            ),
            "suspicious": (
                "Quantum computing uses quantum mechanical phenomena such as superposition and "
                "entanglement to perform computation. Quantum computers are fundamentally "
                "different from classical computers and can solve certain problems exponentially "
                "faster."
            ),
            "is_plagiarized": False,
        },
        {
            "type": "direct_copy",
            "source": (
                "The Internet of Things describes physical objects with sensors, processing "
                "ability, software and other technologies that connect and exchange data with "
                "other devices and systems over the Internet or other communications networks."
            ),
            "suspicious": (
                "The Internet of Things describes physical objects with sensors, processing "
                "ability, software and other technologies that connect and exchange data with "
                "other devices and systems over the Internet or other communications networks."
            ),
            "is_plagiarized": True,
        },
        {
            "type": "paraphrase",
            "source": (
                "Blockchain is a system of recording information in a way that makes it "
                "difficult or impossible to change, hack, or cheat the system. A blockchain "
                "is essentially a digital ledger of transactions that is duplicated and "
                "distributed across the entire network of computer systems on the blockchain."
            ),
            "suspicious": (
                "A blockchain is a method of storing data that is highly resistant to "
                "modification, hacking, or fraud. It functions as a digital transaction "
                "record that is copied and spread across all computers participating in "
                "the network, making it transparent and tamper-proof."
            ),
            "is_plagiarized": True,
        },
        {
            "type": "clean",
            "source": (
                "Photosynthesis is the process used by plants, algae and certain bacteria "
                "to harness energy from sunlight and turn it into chemical energy. This "
                "stored chemical energy is used for the organisms activities."
            ),
            "suspicious": (
                "Cloud computing is the on-demand availability of computer system resources, "
                "especially data storage and computing power, without direct active management "
                "by the user. Large clouds often have functions distributed over multiple "
                "locations, each location being a data center."
            ),
            "is_plagiarized": False,
        },
        {
            "type": "mosaic",
            "source": (
                "Cybersecurity is the practice of protecting systems, networks, and programs "
                "from digital attacks. These cyberattacks are usually aimed at accessing, "
                "changing, or destroying sensitive information, extorting money from users, "
                "or interrupting normal business processes."
            ),
            "suspicious": (
                "The practice of protecting systems and networks from digital attacks is "
                "known as cybersecurity. Malicious actors typically target sensitive "
                "information or seek to extort money, while also attempting to disrupt "
                "normal operations of businesses and organizations."
            ),
            "is_plagiarized": True,
        },
        {
            "type": "clean",
            "source": (
                "Augmented reality is an interactive experience that combines the real world "
                "and computer-generated content. The content can span multiple sensory "
                "modalities, including visual, auditory, haptic, somatosensory and olfactory."
            ),
            "suspicious": (
                "The theory of evolution by natural selection was first formulated in "
                "Charles Darwins book On the Origin of Species. Natural selection is the "
                "process by which organisms that are better adapted to their environment "
                "tend to survive and produce more offspring."
            ),
            "is_plagiarized": False,
        },
        {
            "type": "direct_copy",
            "source": (
                "Convolutional neural networks are a class of artificial neural network "
                "most commonly applied to analyze visual imagery. They are also known as "
                "shift invariant or space invariant artificial neural networks, based on "
                "the shared-weight architecture of the convolution kernels that scan the "
                "hidden layers."
            ),
            "suspicious": (
                "Convolutional neural networks are a class of artificial neural network "
                "most commonly applied to analyze visual imagery. They are also known as "
                "shift invariant or space invariant artificial neural networks, based on "
                "the shared-weight architecture of the convolution kernels that scan the "
                "hidden layers."
            ),
            "is_plagiarized": True,
        },
        {
            "type": "paraphrase",
            "source": (
                "Reinforcement learning is an area of machine learning concerned with how "
                "intelligent agents ought to take actions in an environment in order to "
                "maximize the notion of cumulative reward. Reinforcement learning is one "
                "of the three basic machine learning paradigms, alongside supervised "
                "learning and unsupervised learning."
            ),
            "suspicious": (
                "Reinforcement learning is a branch of machine learning that studies how "
                "an agent should act within an environment to maximize its total reward. "
                "It represents one of the three core paradigms of machine learning, "
                "together with supervised and unsupervised approaches."
            ),
            "is_plagiarized": True,
        },
        {
            "type": "clean",
            "source": (
                "The human genome contains approximately 3 billion base pairs of DNA "
                "organized into 23 chromosome pairs. The complete set of genetic "
                "instructions for making a human being is encoded within this sequence."
            ),
            "suspicious": (
                "Graph neural networks are a type of neural network that operates on "
                "graph-structured data. They learn representations of nodes and edges "
                "by aggregating information from local neighborhoods in the graph structure."
            ),
            "is_plagiarized": False,
        },
    ]

    def parse_pan_xml(self, xml_file: str) -> List[Dict]:
        """Parse a PAN XML annotation file.

        Expected XML structure::

            <document reference="suspicious-doc.txt">
              <feature name="plagiarism"
                       this_offset="0" this_length="100"
                       source_reference="source-doc.txt"
                       source_offset="0" source_length="100"/>
            </document>

        Args:
            xml_file: Path to the PAN XML annotation file.

        Returns:
            List of annotation dicts with keys: source_reference,
            this_offset, this_length, source_offset, source_length.
        """
        annotations: List[Dict] = []
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            doc_reference = root.get("reference", "")
            for feature in root.findall(".//feature[@name='plagiarism']"):
                annotations.append(
                    {
                        "document_reference": doc_reference,
                        "source_reference": feature.get("source_reference", ""),
                        "this_offset": int(feature.get("this_offset", 0)),
                        "this_length": int(feature.get("this_length", 0)),
                        "source_offset": int(feature.get("source_offset", 0)),
                        "source_length": int(feature.get("source_length", 0)),
                    }
                )
        except ET.ParseError as exc:
            logger.error("Failed to parse XML file %s: %s", xml_file, exc)
        except FileNotFoundError:
            logger.error("XML file not found: %s", xml_file)
        return annotations

    def download_corpus(self, version: str = "pan11", output_dir: str = "data/pan_corpus") -> str:
        """Provide guidance for obtaining the official PAN corpus.

        The PAN shared task datasets are freely available for research but
        require registration at https://pan.webis.de/.  This method prints
        download instructions and falls back to creating a sample corpus.

        Args:
            version: PAN shared task year/version, e.g. ``"pan11"`` or ``"pan13"``.
            output_dir: Directory where the corpus should be stored.

        Returns:
            Path to the output directory.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print(f"PAN Corpus Download Guide (version: {version})")
        print("=" * 60)
        print()
        print("Official PAN datasets are available at:")
        print("  https://pan.webis.de/")
        print()
        print("Steps to obtain the official corpus:")
        print("  1. Visit https://pan.webis.de/data.html")
        print(f"  2. Search for the '{version}' plagiarism detection task")
        print("  3. Register for a free account if required")
        print("  4. Download the corpus files")
        print(f"  5. Extract to: {output_path.resolve()}")
        print()
        print("Corpus file structure expected:")
        print(f"  {output_dir}/")
        print("    source-documents/    # source text files")
        print("    suspicious-documents/ # suspicious text files")
        print("    pairs                # document pair annotations")
        print()
        print("Creating a sample corpus for testing purposes...")
        sample_dir = self.create_sample_corpus(str(output_path / "sample"))
        print(f"Sample corpus created at: {sample_dir}")
        return str(output_path)

    def load_corpus(self, corpus_dir: str) -> List[Dict]:
        """Load document pairs from a PAN-style corpus directory.

        Expects the following structure inside *corpus_dir*::

            source-documents/       .txt files
            suspicious-documents/   .txt files
            *.xml                   PAN annotation files (one per suspicious doc)

        If XML annotations are absent every pair is labelled as not plagiarised.

        Args:
            corpus_dir: Root directory of the corpus.

        Returns:
            List of dicts with keys: ``text1``, ``text2``, ``label``,
            ``source_file``, ``suspicious_file``.
        """
        corpus_path = Path(corpus_dir)
        pairs: List[Dict] = []

        source_dir = corpus_path / "source-documents"
        suspicious_dir = corpus_path / "suspicious-documents"

        if not source_dir.exists() or not suspicious_dir.exists():
            logger.warning(
                "Expected sub-directories 'source-documents' and "
                "'suspicious-documents' not found in %s. "
                "Attempting flat directory load.",
                corpus_dir,
            )
            return self._load_flat_corpus(corpus_dir)

        # Build a lookup for annotation XML files
        annotation_map: Dict[str, List[Dict]] = {}
        for xml_file in corpus_path.glob("*.xml"):
            annotations = self.parse_pan_xml(str(xml_file))
            if annotations:
                doc_ref = annotations[0]["document_reference"]
                annotation_map[doc_ref] = annotations

        # If a pairs index file exists, use it for complete pair coverage
        pairs_index_file = corpus_path / "pairs"
        if pairs_index_file.exists():
            return self._load_from_pairs_index(
                pairs_index_file, source_dir, suspicious_dir, annotation_map
            )

        suspicious_files = sorted(suspicious_dir.glob("*.txt"))
        for suspicious_file in suspicious_files:
            annotations = annotation_map.get(suspicious_file.name, [])
            is_plagiarized = len(annotations) > 0

            try:
                suspicious_text = suspicious_file.read_text(encoding="utf-8")
            except OSError as exc:
                logger.error("Cannot read %s: %s", suspicious_file, exc)
                continue

            if annotations:
                for ann in annotations:
                    source_file = source_dir / ann["source_reference"]
                    if not source_file.exists():
                        continue
                    try:
                        source_text = source_file.read_text(encoding="utf-8")
                    except OSError as exc:
                        logger.error("Cannot read %s: %s", source_file, exc)
                        continue
                    pairs.append(
                        {
                            "text1": source_text,
                            "text2": suspicious_text,
                            "label": 1,
                            "source_file": str(source_file),
                            "suspicious_file": str(suspicious_file),
                        }
                    )
            else:
                # No annotations → try to match by name convention
                source_file = source_dir / suspicious_file.name
                if source_file.exists():
                    try:
                        source_text = source_file.read_text(encoding="utf-8")
                        pairs.append(
                            {
                                "text1": source_text,
                                "text2": suspicious_text,
                                "label": 0,
                                "source_file": str(source_file),
                                "suspicious_file": str(suspicious_file),
                            }
                        )
                    except OSError as exc:
                        logger.error("Cannot read %s: %s", source_file, exc)

        logger.info("Loaded %d document pairs from %s", len(pairs), corpus_dir)
        return pairs

    def _load_from_pairs_index(
        self,
        pairs_index_file: Path,
        source_dir: Path,
        suspicious_dir: Path,
        annotation_map: Dict[str, List[Dict]],
    ) -> List[Dict]:
        """Load pairs using the tab-separated ``pairs`` index file.

        The index format (written by :meth:`create_sample_corpus`) is::

            # source_document  suspicious_document  label  type
            source-doc00001.txt  suspicious-doc00001.txt  1  direct_copy

        Args:
            pairs_index_file: Path to the ``pairs`` file.
            source_dir: Directory containing source documents.
            suspicious_dir: Directory containing suspicious documents.
            annotation_map: Pre-built annotation lookup (doc_ref → annotations).

        Returns:
            List of pair dicts.
        """
        pairs: List[Dict] = []
        for line in pairs_index_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            src_name, sus_name, label_str = parts[0], parts[1], parts[2]
            source_file = source_dir / src_name
            suspicious_file = suspicious_dir / sus_name
            if not source_file.exists() or not suspicious_file.exists():
                logger.warning("Skipping missing files: %s / %s", src_name, sus_name)
                continue
            try:
                source_text = source_file.read_text(encoding="utf-8")
                suspicious_text = suspicious_file.read_text(encoding="utf-8")
            except OSError as exc:
                logger.error("Cannot read pair files: %s", exc)
                continue
            pairs.append(
                {
                    "text1": source_text,
                    "text2": suspicious_text,
                    "label": int(label_str),
                    "source_file": str(source_file),
                    "suspicious_file": str(suspicious_file),
                }
            )
        return pairs

    def _load_flat_corpus(self, corpus_dir: str) -> List[Dict]:
        """Fallback loader for flat directory layouts."""
        corpus_path = Path(corpus_dir)
        txt_files = sorted(corpus_path.glob("*.txt"))
        pairs: List[Dict] = []
        for i in range(0, len(txt_files) - 1, 2):
            try:
                text1 = txt_files[i].read_text(encoding="utf-8")
                text2 = txt_files[i + 1].read_text(encoding="utf-8")
                pairs.append(
                    {
                        "text1": text1,
                        "text2": text2,
                        "label": 0,
                        "source_file": str(txt_files[i]),
                        "suspicious_file": str(txt_files[i + 1]),
                    }
                )
            except OSError as exc:
                logger.error("Cannot read file pair: %s", exc)
        return pairs

    def get_corpus_statistics(self, corpus_dir: str) -> Dict:
        """Compute statistics for a loaded corpus.

        Args:
            corpus_dir: Root directory of the corpus.

        Returns:
            Dict with: total_pairs, plagiarized_pairs, clean_pairs,
            avg_text_length, plagiarism_rate.
        """
        pairs = self.load_corpus(corpus_dir)
        if not pairs:
            return {
                "total_pairs": 0,
                "plagiarized_pairs": 0,
                "clean_pairs": 0,
                "avg_text_length": 0.0,
                "plagiarism_rate": 0.0,
            }

        plagiarized = sum(1 for p in pairs if p["label"] == 1)
        clean = len(pairs) - plagiarized
        avg_len = sum(len(p["text1"]) + len(p["text2"]) for p in pairs) / (2 * len(pairs))

        return {
            "total_pairs": len(pairs),
            "plagiarized_pairs": plagiarized,
            "clean_pairs": clean,
            "avg_text_length": round(avg_len, 2),
            "plagiarism_rate": round(plagiarized / len(pairs), 4),
        }

    def create_sample_corpus(self, output_dir: str = "data/pan_corpus/sample") -> str:
        """Create a sample PAN-style corpus with XML annotations for testing.

        Generates source documents, suspicious documents, and corresponding
        PAN XML annotation files covering direct copy, paraphrase, and mosaic
        plagiarism types, as well as clean (non-plagiarised) pairs.

        Args:
            output_dir: Directory to write the sample corpus into.

        Returns:
            Path to the created corpus directory.
        """
        output_path = Path(output_dir)
        source_dir = output_path / "source-documents"
        suspicious_dir = output_path / "suspicious-documents"
        source_dir.mkdir(parents=True, exist_ok=True)
        suspicious_dir.mkdir(parents=True, exist_ok=True)

        for idx, pair in enumerate(self._SAMPLE_PAIRS, start=1):
            source_filename = f"source-doc{idx:05d}.txt"
            suspicious_filename = f"suspicious-doc{idx:05d}.txt"

            source_text = pair["source"]
            suspicious_text = pair["suspicious"]

            (source_dir / source_filename).write_text(source_text, encoding="utf-8")
            (suspicious_dir / suspicious_filename).write_text(suspicious_text, encoding="utf-8")

            # Write PAN XML annotation
            xml_lines = [
                '<?xml version="1.0" encoding="UTF-8"?>',
                f'<document reference="{suspicious_filename}">',
            ]
            if pair["is_plagiarized"]:
                xml_lines.append(
                    f'  <feature name="plagiarism"'
                    f' this_offset="0"'
                    f' this_length="{len(suspicious_text)}"'
                    f' source_reference="{source_filename}"'
                    f' source_offset="0"'
                    f' source_length="{len(source_text)}"'
                    f' type="{pair["type"]}"/>'
                )
            xml_lines.append("</document>")
            xml_content = "\n".join(xml_lines)
            (output_path / f"suspicious-doc{idx:05d}.xml").write_text(
                xml_content, encoding="utf-8"
            )

        # Write a pairs index file for convenience
        pairs_lines = ["# source_document\tsuspicious_document\tlabel\ttype"]
        for idx, pair in enumerate(self._SAMPLE_PAIRS, start=1):
            label = "1" if pair["is_plagiarized"] else "0"
            pairs_lines.append(
                f"source-doc{idx:05d}.txt\t"
                f"suspicious-doc{idx:05d}.txt\t"
                f"{label}\t{pair['type']}"
            )
        (output_path / "pairs").write_text("\n".join(pairs_lines), encoding="utf-8")

        logger.info(
            "Created sample corpus with %d pairs at %s",
            len(self._SAMPLE_PAIRS),
            output_path,
        )
        print(f"Sample corpus created: {len(self._SAMPLE_PAIRS)} pairs ({output_path})")
        return str(output_path)
