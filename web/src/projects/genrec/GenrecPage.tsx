import { useEffect } from "react";
import { useSearchParams } from "react-router-dom";

import { useJson } from "../../lib/api";
import Explorer, { API } from "./Explorer";

export default function GenrecPage() {
  const [params, setParams] = useSearchParams();
  const userParam = params.get("user");
  const users = useJson<number[]>(`${API}/users?limit=1000`);

  // no user in the URL yet: start with a random one (the URL keeps it, so a view can be shared)
  useEffect(() => {
    if (!userParam && users.data?.length) {
      setParams({ user: String(users.data[Math.floor(Math.random() * users.data.length)]) }, { replace: true });
    }
  }, [userParam, users.data, setParams]);

  return (
    <div>
      <Explorer userId={userParam ? Number(userParam) : null} users={users.data}
                onPick={(id) => setParams({ user: String(id) })} />
    </div>
  );
}
