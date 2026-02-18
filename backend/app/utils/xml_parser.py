from __future__ import annotations

from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Iterable, Iterator, List
from xml.etree import ElementTree as ET

from langchain_core.documents import Document


@dataclass(frozen=True)
class TopicMetadata:
	title: str
	url: str | None
	language: str | None
	topic_id: str | None
	groups: tuple[str, ...]
	aliases: tuple[str, ...]
	source_file: str


class XMLParser:
	"""Loads MedlinePlus XML topics and converts them into LangChain documents."""

	def __init__(self, source: str | Path):
		base_path = Path(source)
		if base_path.is_dir():
			self._sources = sorted(base_path.glob("*.xml"))
		else:
			self._sources = [base_path]

		if not self._sources:
			raise FileNotFoundError(f"No XML files found under {base_path!s}")

	def load_documents(self) -> List[Document]:
		documents: List[Document] = []
		for xml_file in self._sources:
			for document in self._iter_documents(xml_file):
				documents.append(document)

		if not documents:
			raise ValueError("XML parser did not yield any documents")

		return documents

	def _iter_documents(self, xml_file: Path) -> Iterator[Document]:
		for _, element in ET.iterparse(xml_file, events=("end",)):
			if element.tag != "health-topic":
				continue

			metadata = self._extract_metadata(element, xml_file)
			page_content = self._build_page_content(element, metadata)

			yield Document(page_content=page_content, metadata=metadata.__dict__)
			element.clear()

	def _extract_metadata(self, element: ET.Element, xml_file: Path) -> TopicMetadata:
		groups = tuple(self._collect_text(element, "group"))
		aliases = tuple(self._collect_text(element, "also-called"))

		return TopicMetadata(
			title=element.attrib.get("title", "Untitled"),
			url=element.attrib.get("url"),
			language=element.attrib.get("language"),
			topic_id=element.attrib.get("id"),
			groups=groups,
			aliases=aliases,
			source_file=xml_file.name,
		)

	def _build_page_content(self, element: ET.Element, metadata: TopicMetadata) -> str:
		summary = self._clean(element.findtext("full-summary"))
		groups_line = f"Groups: {', '.join(metadata.groups)}" if metadata.groups else ""
		aliases_line = f"Also called: {', '.join(metadata.aliases)}" if metadata.aliases else ""
		see_also = self._collect_text(element, "see-reference")
		see_also_line = f"See also: {', '.join(see_also)}" if see_also else ""

		sections = [
			metadata.title,
			self._clean(element.attrib.get("meta-desc")),
			summary,
			aliases_line,
			groups_line,
			see_also_line,
		]

		return "\n\n".join(filter(None, sections))

	def _collect_text(self, element: ET.Element, tag_name: str) -> Iterable[str]:
		for child in element.findall(tag_name):
			cleaned = self._clean(child.text)
			if cleaned:
				yield cleaned

	def _clean(self, value: str | None) -> str:
		if not value:
			return ""
		return unescape(value).strip()
