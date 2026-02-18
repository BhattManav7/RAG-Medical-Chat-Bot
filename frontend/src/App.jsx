import Chatbot from "./components/Chatbot";

const styles = {
	app: {
		minHeight: "100vh",
		margin: 0,
		background: "linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%)",
		color: "#f8fafc",
		fontFamily: "'Space Grotesk', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
		display: "flex",
		justifyContent: "center",
		alignItems: "center",
		padding: "2rem",
	},
	panel: {
		width: "min(100%, 960px)",
		backgroundColor: "rgba(15, 23, 42, 0.8)",
		border: "1px solid rgba(148, 163, 184, 0.3)",
		borderRadius: "1.5rem",
		boxShadow: "0 25px 60px rgba(15, 23, 42, 0.65)",
		padding: "2rem",
	},
	title: {
		marginTop: 0,
		marginBottom: "1.5rem",
		fontSize: "1.75rem",
		fontWeight: 600,
		letterSpacing: "0.02em",
	},
	subtitle: {
		marginTop: 0,
		marginBottom: "2rem",
		color: "#94a3b8",
		fontSize: "1rem",
	},
};

const App = () => {
	const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

	return (
		<main style={styles.app}>
			<section style={styles.panel}>
				<h1 style={styles.title}>RAG Medical Chatbot</h1>
				<p style={styles.subtitle}>
					Ask evidence-based medical questions. Responses cite MedlinePlus knowledge base entries so you
					can trace every recommendation back to trusted content.
				</p>
				<Chatbot apiBaseUrl={apiBaseUrl} />
			</section>
		</main>
	);
};

export default App;
