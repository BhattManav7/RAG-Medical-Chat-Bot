const baseStyles = {
	container: {
		display: "flex",
		flexDirection: "column",
		gap: "0.35rem",
		borderRadius: "1rem",
		padding: "0.9rem 1rem",
	},
	role: {
		fontSize: "0.85rem",
		letterSpacing: "0.06em",
		textTransform: "uppercase",
	},
	content: {
		margin: 0,
		lineHeight: 1.5,
		fontSize: "1rem",
	},
	sources: {
		margin: 0,
		color: "#38bdf8",
		fontSize: "0.9rem",
	},
};

const palette = {
	assistant: {
		background: "rgba(15, 118, 110, 0.15)",
		border: "1px solid rgba(16, 185, 129, 0.4)",
		roleColor: "#2dd4bf",
	},
	user: {
		background: "rgba(59, 130, 246, 0.12)",
		border: "1px solid rgba(96, 165, 250, 0.45)",
		roleColor: "#60a5fa",
	},
};

const Message = ({ role, content, sources = [] }) => {
	const scheme = palette[role] ?? palette.assistant;

	return (
		<article
			style={{
				...baseStyles.container,
				background: scheme.background,
				border: scheme.border,
			}}
		>
			<span style={{ ...baseStyles.role, color: scheme.roleColor }}>{role}</span>
			<p style={baseStyles.content}>{content}</p>
			{!!sources.length && (
				<p style={baseStyles.sources}>Sources: {sources.join(", ")}</p>
			)}
		</article>
	);
};

export default Message;
