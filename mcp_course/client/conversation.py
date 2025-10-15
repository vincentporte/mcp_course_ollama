"""Conversation management with MCP tool integration."""

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any
import uuid

from mcp_course.client.integration import ConversationContext, OllamaMCPBridge
from mcp_course.client.prompts import PromptEngineering


@dataclass
class ConversationMetadata:
    """Metadata for conversation tracking."""
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0
    tool_calls_count: int = 0
    participants: list[str] = field(default_factory=lambda: ["user", "assistant"])
    tags: list[str] = field(default_factory=list)


class ConversationManager:
    """
    Manages conversations with MCP tool integration.

    This class provides:
    - Conversation persistence and loading
    - Context management across multiple turns
    - Tool usage tracking and analytics
    - Conversation search and organization
    """

    def __init__(
        self,
        bridge: OllamaMCPBridge,
        storage_path: Path | None = None
    ):
        """Initialize conversation manager."""
        self.bridge = bridge
        self.storage_path = storage_path or Path.home() / ".mcp" / "conversations"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.active_conversations: dict[str, ConversationContext] = {}
        self.conversation_metadata: dict[str, ConversationMetadata] = {}
        self.logger = logging.getLogger("ConversationManager")

        # Load existing conversations
        self._load_conversations()

    def _load_conversations(self) -> None:
        """Load existing conversations from storage."""
        try:
            metadata_file = self.storage_path / "metadata.json"
            if metadata_file.exists():
                with Path.open(metadata_file) as f:
                    metadata_data = json.load(f)

                for conv_id, meta_dict in metadata_data.items():
                    # Convert datetime strings back to datetime objects
                    meta_dict["created_at"] = datetime.fromisoformat(meta_dict["created_at"])
                    meta_dict["updated_at"] = datetime.fromisoformat(meta_dict["updated_at"])

                    self.conversation_metadata[conv_id] = ConversationMetadata(**meta_dict)

                self.logger.info(f"Loaded metadata for {len(self.conversation_metadata)} conversations")

        except Exception as e:
            self.logger.error(f"Error loading conversation metadata: {e}")

    def _save_metadata(self) -> None:
        """Save conversation metadata to storage."""
        try:
            metadata_file = self.storage_path / "metadata.json"

            # Convert metadata to serializable format
            metadata_dict = {}
            for conv_id, metadata in self.conversation_metadata.items():
                meta_dict = asdict(metadata)
                meta_dict["created_at"] = metadata.created_at.isoformat()
                meta_dict["updated_at"] = metadata.updated_at.isoformat()
                metadata_dict[conv_id] = meta_dict

            with Path.open(metadata_file, 'w') as f:
                json.dump(metadata_dict, f, indent=2)

        except Exception as e:
            self.logger.error(f"Error saving conversation metadata: {e}")

    def create_conversation(
        self,
        title: str | None = None,
        tags: list[str] | None = None
    ) -> str:
        """
        Create a new conversation.

        Args:
            title: Conversation title
            tags: Tags for organization

        Returns:
            Conversation ID
        """
        conv_id = str(uuid.uuid4())
        now = datetime.now()

        # Create conversation context
        context = ConversationContext(
            conversation_id=conv_id,
            created_at=now
        )

        # Create metadata
        metadata = ConversationMetadata(
            id=conv_id,
            title=title or f"Conversation {now.strftime('%Y-%m-%d %H:%M')}",
            created_at=now,
            updated_at=now,
            tags=tags or []
        )

        # Store in memory
        self.active_conversations[conv_id] = context
        self.conversation_metadata[conv_id] = metadata

        # Save metadata
        self._save_metadata()

        self.logger.info(f"Created new conversation: {conv_id}")
        return conv_id

    def load_conversation(self, conv_id: str) -> ConversationContext | None:
        """
        Load a conversation from storage.

        Args:
            conv_id: Conversation ID

        Returns:
            ConversationContext if found, None otherwise
        """
        if conv_id in self.active_conversations:
            return self.active_conversations[conv_id]

        try:
            conv_file = self.storage_path / f"{conv_id}.json"
            if conv_file.exists():
                with Path.open(conv_file) as f:
                    conv_data = json.load(f)

                # Reconstruct conversation context
                context = ConversationContext(
                    messages=conv_data.get("messages", []),
                    tool_results=conv_data.get("tool_results", {}),
                    conversation_id=conv_id,
                    created_at=datetime.fromisoformat(conv_data.get("created_at", datetime.now().isoformat()))
                )

                # Cache in memory
                self.active_conversations[conv_id] = context

                self.logger.info(f"Loaded conversation: {conv_id}")
                return context

        except Exception as e:
            self.logger.error(f"Error loading conversation {conv_id}: {e}")

        return None

    def save_conversation(self, conv_id: str) -> bool:
        """
        Save a conversation to storage.

        Args:
            conv_id: Conversation ID

        Returns:
            True if saved successfully
        """
        if conv_id not in self.active_conversations:
            self.logger.error(f"Conversation {conv_id} not found in active conversations")
            return False

        try:
            context = self.active_conversations[conv_id]
            conv_file = self.storage_path / f"{conv_id}.json"

            # Prepare data for serialization
            conv_data = {
                "conversation_id": conv_id,
                "messages": context.messages,
                "tool_results": context.tool_results,
                "created_at": context.created_at.isoformat()
            }

            with Path.open(conv_file, 'w') as f:
                json.dump(conv_data, f, indent=2)

            # Update metadata
            if conv_id in self.conversation_metadata:
                metadata = self.conversation_metadata[conv_id]
                metadata.updated_at = datetime.now()
                metadata.message_count = len(context.messages)
                metadata.tool_calls_count = len(context.tool_results)
                self._save_metadata()

            self.logger.info(f"Saved conversation: {conv_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving conversation {conv_id}: {e}")
            return False

    async def send_message(
        self,
        conv_id: str,
        message: str,
        model: str | None = None,
        auto_save: bool = True,
        system_prompt: str | None = None,
        stream: bool = False
    ) -> dict[str, Any]:
        """
        Send a message in a conversation with tool integration.

        Args:
            conv_id: Conversation ID
            message: User message
            model: Ollama model to use
            auto_save: Whether to auto-save after message
            system_prompt: Optional system prompt
            stream: Whether to stream the response

        Returns:
            Response dictionary with conversation results
        """
        # Load or get conversation
        context = self.load_conversation(conv_id)
        if not context:
            self.logger.error(f"Conversation {conv_id} not found")
            return {"error": f"Conversation {conv_id} not found"}

        try:
            # Send message through bridge
            result = await self.bridge.chat_with_tools(
                message=message,
                model=model,
                context=context,
                system_prompt=system_prompt,
                stream=stream
            )

            # Update active conversation
            self.active_conversations[conv_id] = result["context"]

            # Auto-save if requested
            if auto_save:
                self.save_conversation(conv_id)

            # Add conversation metadata to result
            result["conversation_id"] = conv_id
            result["message_count"] = len(context.messages)

            return result

        except Exception as e:
            self.logger.error(f"Error sending message in conversation {conv_id}: {e}")
            return {"error": str(e), "conversation_id": conv_id}

    async def send_message_with_prompt_template(
        self,
        conv_id: str,
        message: str,
        template_name: str = "conversational",
        model: str | None = None,
        auto_save: bool = True
    ) -> dict[str, Any]:
        """
        Send a message using a specific prompt template.

        Args:
            conv_id: Conversation ID
            message: User message
            template_name: Name of prompt template to use
            model: Ollama model to use
            auto_save: Whether to auto-save after message

        Returns:
            Response dictionary with conversation results
        """


        # Load conversation context
        context = self.load_conversation(conv_id)
        if not context:
            return {"error": f"Conversation {conv_id} not found"}

        try:
            # Create prompt engineering instance
            prompt_eng = PromptEngineering()

            # Get available tools from bridge
            available_tools = list(self.bridge.available_tools.values())

            # Generate enhanced prompt
            enhanced_message = prompt_eng.generate_prompt(
                template_name=template_name,
                user_message=message,
                tools=available_tools,
                conversation_history=context.messages[-5:] if context.messages else None
            )

            # Send the enhanced message
            return await self.send_message(
                conv_id=conv_id,
                message=enhanced_message,
                model=model,
                auto_save=auto_save
            )

        except Exception as e:
            self.logger.error(f"Error sending templated message: {e}")
            return {"error": str(e), "conversation_id": conv_id}

    def get_conversation_analytics(self, conv_id: str) -> dict[str, Any] | None:
        """
        Get analytics for a conversation.

        Args:
            conv_id: Conversation ID

        Returns:
            Analytics dictionary or None if not found
        """
        context = self.active_conversations.get(conv_id)
        if not context:
            context = self.load_conversation(conv_id)

        if not context:
            return None

        # Calculate analytics
        user_messages = [msg for msg in context.messages if msg["role"] == "user"]
        assistant_messages = [msg for msg in context.messages if msg["role"] == "assistant"]
        tool_messages = [msg for msg in context.messages if msg["role"] == "tool"]

        total_chars = sum(len(msg["content"]) for msg in context.messages)
        avg_message_length = total_chars / len(context.messages) if context.messages else 0

        return {
            "conversation_id": conv_id,
            "total_messages": len(context.messages),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "tool_messages": len(tool_messages),
            "tools_used": list(context.tool_results.keys()),
            "unique_tools_count": len(context.tool_results),
            "total_characters": total_chars,
            "average_message_length": round(avg_message_length, 2),
            "conversation_duration": self._calculate_duration(context),
            "tool_usage_frequency": self._calculate_tool_frequency(context)
        }

    def _calculate_duration(self, context: ConversationContext) -> str:
        """Calculate conversation duration."""
        if not context.messages or len(context.messages) < 2:
            return "0 minutes"

        try:
            first_msg = context.messages[0]
            last_msg = context.messages[-1]

            if "timestamp" in first_msg and "timestamp" in last_msg:
                start_time = datetime.fromisoformat(first_msg["timestamp"])
                end_time = datetime.fromisoformat(last_msg["timestamp"])
                duration = end_time - start_time

                minutes = duration.total_seconds() / 60
                if minutes < 1:
                    return "< 1 minute"
                elif minutes < 60:
                    return f"{int(minutes)} minutes"
                else:
                    hours = int(minutes / 60)
                    remaining_minutes = int(minutes % 60)
                    return f"{hours}h {remaining_minutes}m"
        except Exception:
            pass

        return "Unknown"

    def _calculate_tool_frequency(self, context: ConversationContext) -> dict[str, int]:
        """Calculate how frequently each tool was used."""
        tool_frequency = {}

        for msg in context.messages:
            if msg["role"] == "tool":
                try:
                    tool_results = json.loads(msg["content"])
                    if isinstance(tool_results, list):
                        for result in tool_results:
                            if isinstance(result, dict) and "function_name" in result:
                                tool_name = result["function_name"]
                                tool_frequency[tool_name] = tool_frequency.get(tool_name, 0) + 1
                except (json.JSONDecodeError, KeyError):
                    pass

        return tool_frequency

    def export_conversation(
        self,
        conv_id: str,
        format_type: str = "json",
        include_metadata: bool = True
    ) -> str | None:
        """
        Export conversation in specified format.

        Args:
            conv_id: Conversation ID
            format_type: Export format ("json", "markdown", "txt")
            include_metadata: Whether to include metadata

        Returns:
            Exported conversation string or None if error
        """
        context = self.load_conversation(conv_id)
        metadata = self.conversation_metadata.get(conv_id)

        if not context:
            return None

        try:
            if format_type == "json":
                export_data = {
                    "conversation_id": conv_id,
                    "messages": context.messages,
                    "tool_results": context.tool_results
                }

                if include_metadata and metadata:
                    export_data["metadata"] = {
                        "title": metadata.title,
                        "created_at": metadata.created_at.isoformat(),
                        "updated_at": metadata.updated_at.isoformat(),
                        "tags": metadata.tags
                    }

                return json.dumps(export_data, indent=2)

            elif format_type == "markdown":
                lines = []

                if include_metadata and metadata:
                    lines.append(f"# {metadata.title}")
                    lines.append(f"**Created:** {metadata.created_at}")
                    lines.append(f"**Tags:** {', '.join(metadata.tags)}")
                    lines.append("")

                for msg in context.messages:
                    role = msg["role"].title()
                    content = msg["content"]
                    timestamp = msg.get("timestamp", "")

                    lines.append(f"## {role}")
                    if timestamp:
                        lines.append(f"*{timestamp}*")
                    lines.append("")
                    lines.append(content)
                    lines.append("")

                return "\n".join(lines)

            elif format_type == "txt":
                lines = []

                if include_metadata and metadata:
                    lines.append(f"Conversation: {metadata.title}")
                    lines.append(f"Created: {metadata.created_at}")
                    lines.append("-" * 50)
                    lines.append("")

                for msg in context.messages:
                    role = msg["role"].upper()
                    content = msg["content"]
                    timestamp = msg.get("timestamp", "")

                    header = f"[{role}]"
                    if timestamp:
                        header += f" {timestamp}"

                    lines.append(header)
                    lines.append(content)
                    lines.append("")

                return "\n".join(lines)

        except Exception as e:
            self.logger.error(f"Error exporting conversation {conv_id}: {e}")

        return None

    def list_conversations(
        self,
        tag: str | None = None,
        limit: int | None = None
    ) -> list[ConversationMetadata]:
        """
        List conversations with optional filtering.

        Args:
            tag: Filter by tag
            limit: Maximum number of conversations to return

        Returns:
            List of conversation metadata
        """
        conversations = list(self.conversation_metadata.values())

        # Filter by tag if specified
        if tag:
            conversations = [
                conv for conv in conversations
                if tag in conv.tags
            ]

        # Sort by updated_at descending
        conversations.sort(key=lambda x: x.updated_at, reverse=True)

        # Apply limit if specified
        if limit:
            conversations = conversations[:limit]

        return conversations

    def delete_conversation(self, conv_id: str) -> bool:
        """
        Delete a conversation.

        Args:
            conv_id: Conversation ID

        Returns:
            True if deleted successfully
        """
        try:
            # Remove from memory
            if conv_id in self.active_conversations:
                del self.active_conversations[conv_id]

            if conv_id in self.conversation_metadata:
                del self.conversation_metadata[conv_id]

            # Remove file
            conv_file = self.storage_path / f"{conv_id}.json"
            if conv_file.exists():
                conv_file.unlink()

            # Save updated metadata
            self._save_metadata()

            self.logger.info(f"Deleted conversation: {conv_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error deleting conversation {conv_id}: {e}")
            return False

    def get_conversation_summary(self, conv_id: str) -> dict[str, Any] | None:
        """
        Get summary information for a conversation.

        Args:
            conv_id: Conversation ID

        Returns:
            Summary dictionary or None if not found
        """
        if conv_id not in self.conversation_metadata:
            return None

        metadata = self.conversation_metadata[conv_id]
        context = self.active_conversations.get(conv_id)

        summary = {
            "id": conv_id,
            "title": metadata.title,
            "created_at": metadata.created_at.isoformat(),
            "updated_at": metadata.updated_at.isoformat(),
            "message_count": metadata.message_count,
            "tool_calls_count": metadata.tool_calls_count,
            "tags": metadata.tags
        }

        if context:
            # Add recent messages preview
            recent_messages = context.messages[-3:] if context.messages else []
            summary["recent_messages"] = [
                {
                    "role": msg["role"],
                    "content": msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"],
                    "timestamp": msg.get("timestamp", "")
                }
                for msg in recent_messages
            ]

            # Add tools used
            summary["tools_used"] = list(context.tool_results.keys())

        return summary

    def search_conversations(
        self,
        query: str,
        search_content: bool = True
    ) -> list[str]:
        """
        Search conversations by title or content.

        Args:
            query: Search query
            search_content: Whether to search message content

        Returns:
            List of matching conversation IDs
        """
        matching_conversations = []
        query_lower = query.lower()

        for conv_id, metadata in self.conversation_metadata.items():
            # Search in title and tags
            if (query_lower in metadata.title.lower() or
                any(query_lower in tag.lower() for tag in metadata.tags)):
                matching_conversations.append(conv_id)
                continue

            # Search in content if requested
            if search_content:
                context = self.active_conversations.get(conv_id)
                if not context:
                    context = self.load_conversation(conv_id)

                if context:
                    for message in context.messages:
                        if query_lower in message.get("content", "").lower():
                            matching_conversations.append(conv_id)
                            break

        return matching_conversations
