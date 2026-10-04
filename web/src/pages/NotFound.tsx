import { Link } from "react-router-dom";

export default function NotFound() {
  return (
    <div className="page-head">
      <h1>Page not found</h1>
      <p>
        That page doesn't exist. <Link to="/">Back to the projects</Link>.
      </p>
    </div>
  );
}
