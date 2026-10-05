import { useEffect } from "react";
import { useSearchParams } from "react-router-dom";

import { useJson } from "../../lib/api";
import Explorer, { API } from "./Explorer";
import TryYourOwn from "./TryYourOwn";

export default function GenrecPage() {
  const [params, setParams] = useSearchParams();
  const tryMode = params.get("mode") === "try";
  const userParam = params.get("user");
  const users = useJson<number[]>(`${API}/users?limit=1000`);

  // test users tab with no user in the URL yet: start with a random one (the URL keeps it, so a view can be shared)
  useEffect(() => {
    if (!tryMode && !userParam && users.data?.length) {
      setParams({ user: String(users.data[Math.floor(Math.random() * users.data.length)]) }, { replace: true });
    }
  }, [tryMode, userParam, users.data, setParams]);

  return (
    <div>
      <nav className="tabs" aria-label="GenRec views">
        <button className={!tryMode ? "active" : ""} onClick={() => setParams({})}>Test users</button>
        <button className={tryMode ? "active" : ""} onClick={() => setParams({ mode: "try" })}>Try your own</button>
      </nav>
      {tryMode
        ? <TryYourOwn />
        : <Explorer userId={userParam ? Number(userParam) : null} users={users.data}
                    onPick={(id) => setParams({ user: String(id) })} />}
    </div>
  );
}
