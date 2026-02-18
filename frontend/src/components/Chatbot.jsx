import { useMemo, useState } from "react";
import Message from "./Message";

const styles = {
	wrapper: {
		display: "flex",
		flexDirection: "column",
		gap: "1rem",
	},
	conversation: {
		backgroundColor: "rgba(15, 23, 42, 0.95)",
		border: "1px solid rgba(148, 163, 184, 0.3)",
		borderRadius: "1rem",
		padding: "1.25rem",
		maxHeight: "420px",
		overflowY: "auto",
		display: "flex",
		flexDirection: "column",
		gap: "0.75rem",
	},
	form: {
		display: "flex",
		gap: "0.75rem",
	},
	input: {
		flex: 1,
		padding: "0.85rem 1.1rem",
		borderRadius: "999px",
		border: "1px solid rgba(148, 163, 184, 0.4)",
		backgroundColor: "rgba(15, 23, 42, 0.65)",
		color: "#f8fafc",
		fontSize: "1rem",
	},
	button: {
		borderRadius: "999px",
		border: "none",
		padding: "0 1.5rem",
		fontWeight: 600,
		fontSize: "1rem",
		cursor: "pointer",
		background: "linear-gradient(135deg, #38bdf8 0%, #0ea5e9 100%)",
		color: "#0f172a",
		transition: "opacity 0.2s ease",
	},
	buttonDisabled: {
		opacity: 0.5,
		cursor: "not-allowed",
	},
	helper: {
		color: "#94a3b8",
		fontSize: "0.95rem",
	},
	error: {
		color: "#f87171",
		fontSize: "0.95rem",
	},
};

const Chatbot = ({ apiBaseUrl }) => {
	const [messages, setMessages] = useState([
		{
			id: "assistant-welcome",
			role: "assistant",
			content: "Hi! I’m your MedlinePlus-guided medical assistant. Ask me anything about symptoms, tests, or conditions.",
			sources: [],
		},
	]);
	const [input, setInput] = useState("");
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState("");

	const endpoint = useMemo(() => `${apiBaseUrl.replace(/\/$/, "")}/chat/`, [apiBaseUrl]);

	const submitQuestion = async (event) => {
		event.preventDefault();
		const question = input.trim();
		if (!question || loading) {
			return;
		}

		const userMessage = {
			id: `user-${Date.now()}`,
			role: "user",
			content: question,
			sources: [],
		};

		setMessages((prev) => [...prev, userMessage]);
		setInput("");
		setError("");
		setLoading(true);

		try {
			const response = await fetch(endpoint, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ question }),
			});

			if (!response.ok) {
				const message = await response.text();
				throw new Error(message || "Request failed");
			}

			const json = await response.json();
			const answer = (json.answer ?? "No answer provided.").trim();
			const assistantMessage = {
				id: `assistant-${Date.now()}`,
				role: "assistant",
				content: answer,
				sources: Array.isArray(json.sources) ? json.sources : [],
			};

			setMessages((prev) => [...prev, assistantMessage]);
		} catch (err) {
			setError(err.message || "Something went wrong. Please try again.");
		} finally {
			setLoading(false);
		}
	};

	return (
		<div style={styles.wrapper}>
			<div style={styles.conversation}>
				{messages.map((message) => (
					<Message key={message.id} {...message} />
				))}
			</div>

			<form style={styles.form} onSubmit={submitQuestion}>
				<input
					style={styles.input}
					type="text"
					name="question"
					placeholder="Ask a medical question..."
					value={input}
					onChange={(event) => setInput(event.target.value)}
					disabled={loading}
				/>
				<button
					type="submit"
					style={{ ...styles.button, ...(loading ? styles.buttonDisabled : {}) }}
					disabled={loading}
				>
					{loading ? "Thinking..." : "Ask"}
				</button>
			</form>

			{error ? (
				<p style={styles.error}>{error}</p>
			) : (
				<p style={styles.helper}>Responses cite top MedlinePlus topics pulled from your XML corpus.</p>
			)}
		</div>
	);
};

export default Chatbot;
