import React, { useState, useEffect, useCallback } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import "./AdminDashboard.css";

const API_BASE = "http://localhost:3001";

const authHeader = () => ({
  Authorization: `Bearer ${localStorage.getItem("adminToken")}`,
});

function AdminDashboard({ onAdminLogout }) {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState("overview");

  // ── Data state ──────────────────────────────────────────
  const [stats, setStats]         = useState(null);
  const [users, setUsers]         = useState([]);
  const [auctions, setAuctions]   = useState([]);

  // ── Loading / feedback ──────────────────────────────────
  const [loadingStats,    setLoadingStats]    = useState(false);
  const [loadingUsers,    setLoadingUsers]    = useState(false);
  const [loadingAuctions, setLoadingAuctions] = useState(false);
  const [error,      setError]      = useState(null);
  const [successMsg, setSuccessMsg] = useState(null);

  // ── Auction form ────────────────────────────────────────
  const [auctionForm, setAuctionForm] = useState({ name: "", description: "", quantity: "", price: "" });
  const [addingAuction,   setAddingAuction]   = useState(false);
  const [showAuctionForm, setShowAuctionForm] = useState(false);

  // ── UI helpers ──────────────────────────────────────────
  const [userSearch,    setUserSearch]    = useState("");
  const [confirmDelete, setConfirmDelete] = useState(null);
  const [togglingRole,  setTogglingRole]  = useState(null); // userId being toggled

  // ── Auth guard ──────────────────────────────────────────
  useEffect(() => {
    if (!localStorage.getItem("adminToken")) navigate("/");
  }, [navigate]);

  // ── Flash helper ────────────────────────────────────────
  const flash = useCallback((msg, isError = false) => {
    if (isError) { setError(msg); setTimeout(() => setError(null), 4000); }
    else { setSuccessMsg(msg); setTimeout(() => setSuccessMsg(null), 3000); }
  }, []);

  // ── Fetchers ────────────────────────────────────────────
  const fetchStats = useCallback(async () => {
    setLoadingStats(true);
    try {
      const { data } = await axios.get(`${API_BASE}/admin/stats`, { headers: authHeader() });
      setStats(data);
    } catch (e) {
      flash(e.response?.data?.error || "Failed to load stats", true);
    } finally { setLoadingStats(false); }
  }, [flash]);

  const fetchUsers = useCallback(async () => {
    setLoadingUsers(true);
    try {
      const { data } = await axios.get(`${API_BASE}/admin/users`, { headers: authHeader() });
      setUsers(data.users || []);
    } catch (e) {
      flash(e.response?.data?.error || "Failed to load users", true);
    } finally { setLoadingUsers(false); }
  }, [flash]);

  const fetchAuctions = useCallback(async () => {
    setLoadingAuctions(true);
    try {
      const { data } = await axios.get(`${API_BASE}/admin/auctions`, { headers: authHeader() });
      setAuctions(data.auctions || []);
    } catch (e) {
      flash(e.response?.data?.error || "Failed to load auctions", true);
    } finally { setLoadingAuctions(false); }
  }, [flash]);

  useEffect(() => {
    fetchStats();
    fetchUsers();
    fetchAuctions();
  }, [fetchStats, fetchUsers, fetchAuctions]);

  // ── Delete user ─────────────────────────────────────────
  const handleDeleteUser = async (id) => {
    try {
      await axios.delete(`${API_BASE}/admin/users/${id}`, { headers: authHeader() });
      setUsers((p) => p.filter((u) => u._id !== id));
      flash("✅ User deleted");
      fetchStats();
    } catch (e) {
      flash(e.response?.data?.error || "Failed to delete user", true);
    } finally { setConfirmDelete(null); }
  };

  // ── Toggle role ─────────────────────────────────────────
  const handleToggleRole = async (user) => {
    const newRole = user.role === "admin" ? "user" : "admin";
    setTogglingRole(user._id);
    try {
      const { data } = await axios.patch(
        `${API_BASE}/admin/users/${user._id}/role`,
        { role: newRole },
        { headers: authHeader() }
      );
      setUsers((p) => p.map((u) => (u._id === user._id ? data.user : u)));
      flash(`✅ ${user.username} is now ${newRole}`);
    } catch (e) {
      flash(e.response?.data?.error || "Failed to update role", true);
    } finally { setTogglingRole(null); }
  };

  // ── Delete auction ──────────────────────────────────────
  const handleDeleteAuction = async (id) => {
    try {
      await axios.delete(`${API_BASE}/admin/auctions/${id}`, { headers: authHeader() });
      setAuctions((p) => p.filter((a) => a._id !== id));
      flash("✅ Auction removed");
      fetchStats();
    } catch (e) {
      flash(e.response?.data?.error || "Failed to delete auction", true);
    } finally { setConfirmDelete(null); }
  };

  // ── Add auction ─────────────────────────────────────────
  const handleAddAuction = async (e) => {
    e.preventDefault();
    setAddingAuction(true);
    try {
      const { data } = await axios.post(`${API_BASE}/admin/auctions`, auctionForm, { headers: authHeader() });
      setAuctions((p) => [data.auction, ...p]);
      setAuctionForm({ name: "", description: "", quantity: "", price: "" });
      setShowAuctionForm(false);
      flash("✅ Auction added");
      fetchStats();
    } catch (e) {
      flash(e.response?.data?.error || "Failed to add auction", true);
    } finally { setAddingAuction(false); }
  };

  // ── Logout ──────────────────────────────────────────────
  const handleLogout = () => {
    localStorage.removeItem("adminToken");
    localStorage.removeItem("adminUser");
    if (typeof onAdminLogout === "function") onAdminLogout();
    navigate("/");
  };

  // ── Derived ─────────────────────────────────────────────
  const filteredUsers = users.filter(
    (u) =>
      u.username?.toLowerCase().includes(userSearch.toLowerCase()) ||
      u.email?.toLowerCase().includes(userSearch.toLowerCase())
  );

  const statCards = [
    { label: "Total Users",      value: stats?.totalUsers      ?? "—", icon: "👥", color: "#4A90E2" },
    { label: "Total Auctions",   value: stats?.totalAuctions   ?? "—", icon: "🏷️", color: "#27ae60" },
    { label: "Active Listings",  value: stats?.activeListings  ?? "—", icon: "✅", color: "#f39c12" },
    { label: "Avg Price",        value: stats?.avgPrice != null ? `₹${Number(stats.avgPrice).toLocaleString()}` : "—", icon: "💰", color: "#8e44ad" },
    { label: "New Users (7d)",   value: stats?.newUsers        ?? "—", icon: "🆕", color: "#e74c3c" },
  ];

  // ── Render ──────────────────────────────────────────────
  return (
    <div className="admin-dashboard">
      {/* ── Sidebar ─────────────────────────────── */}
      <aside className="admin-sidebar">
        <div className="sidebar-brand">
          <span className="brand-icon">🌾</span>
          <span className="brand-text">SmartCrop</span>
          <span className="brand-badge">Admin</span>
        </div>

        <nav className="sidebar-nav">
          {[
            { id: "overview", icon: "📊", label: "Overview"  },
            { id: "users",    icon: "👥", label: "Users"     },
            { id: "auctions", icon: "🏷️", label: "Auctions"  },
          ].map(({ id, icon, label }) => (
            <button
              key={id}
              className={`sidebar-nav-btn${activeTab === id ? " active" : ""}`}
              onClick={() => setActiveTab(id)}
              id={`admin-nav-${id}`}
            >
              <span className="nav-icon">{icon}</span>
              <span>{label}</span>
            </button>
          ))}
        </nav>

        <button className="sidebar-logout-btn" onClick={handleLogout} id="admin-logout-btn">
          🚪 Logout
        </button>
      </aside>

      {/* ── Main ────────────────────────────────── */}
      <main className="admin-main">
        <header className="admin-header">
          <div>
            <h1 className="admin-page-title">
              {activeTab === "overview" && "Dashboard Overview"}
              {activeTab === "users"    && "User Management"}
              {activeTab === "auctions" && "Auction Management"}
            </h1>
            <p className="admin-page-sub">
              {activeTab === "overview" && "Platform statistics at a glance"}
              {activeTab === "users"    && `${users.length} registered users`}
              {activeTab === "auctions" && `${auctions.length} total auctions`}
            </p>
          </div>
          <div className="admin-header-right">
            <button className="refresh-btn" onClick={() => { fetchStats(); fetchUsers(); fetchAuctions(); }} title="Refresh all" id="refresh-all-btn">
              🔄
            </button>
            <div className="admin-avatar">A</div>
          </div>
        </header>

        {/* Flash messages */}
        {successMsg && (
          <div className="flash-success" id="admin-flash-success">{successMsg}</div>
        )}
        {error && (
          <div className="flash-error" id="admin-flash-error">
            {error}
            <button className="flash-close" onClick={() => setError(null)}>×</button>
          </div>
        )}

        {/* ════════ OVERVIEW ════════ */}
        {activeTab === "overview" && (
          <section className="overview-section">
            {/* Stats */}
            <div className="stats-grid">
              {statCards.map((s) => (
                <div className="stat-card" key={s.label} style={{ "--accent": s.color }}>
                  <div className="stat-icon">{s.icon}</div>
                  {loadingStats
                    ? <div className="stat-skeleton" />
                    : <div className="stat-value">{s.value}</div>
                  }
                  <div className="stat-label">{s.label}</div>
                </div>
              ))}
            </div>

            {/* Recent users */}
            <div className="overview-panel">
              <h2 className="panel-title">🕑 Recently Registered Users</h2>
              {loadingUsers ? <SkeletonRows n={4} /> : (
                <table className="admin-table">
                  <thead><tr><th>#</th><th>Username</th><th>Email</th><th>Role</th></tr></thead>
                  <tbody>
                    {users.slice(0, 5).map((u, i) => (
                      <tr key={u._id}>
                        <td className="muted">{i + 1}</td>
                        <td className="bold">{u.username}</td>
                        <td className="muted">{u.email}</td>
                        <td><span className={`role-badge role-${u.role}`}>{u.role}</span></td>
                      </tr>
                    ))}
                    {users.length === 0 && <tr><td colSpan={4} className="empty-row">No users yet</td></tr>}
                  </tbody>
                </table>
              )}
            </div>

            {/* Recent auctions */}
            <div className="overview-panel">
              <h2 className="panel-title">🏷️ Recent Auctions</h2>
              {loadingAuctions ? <SkeletonRows n={3} /> : (
                <table className="admin-table">
                  <thead><tr><th>#</th><th>Crop / Item</th><th>Price (₹)</th><th>Qty</th></tr></thead>
                  <tbody>
                    {auctions.slice(0, 5).map((a, i) => (
                      <tr key={a._id}>
                        <td className="muted">{i + 1}</td>
                        <td className="bold">{a.name}</td>
                        <td>{a.price ? `₹${Number(a.price).toLocaleString()}` : "—"}</td>
                        <td>{a.quantity ?? "—"}</td>
                      </tr>
                    ))}
                    {auctions.length === 0 && <tr><td colSpan={4} className="empty-row">No auctions yet</td></tr>}
                  </tbody>
                </table>
              )}
            </div>
          </section>
        )}

        {/* ════════ USERS ════════ */}
        {activeTab === "users" && (
          <section className="content-section">
            <div className="section-toolbar">
              <div className="search-box">
                <span className="search-icon">🔍</span>
                <input
                  type="text"
                  placeholder="Search by username or email…"
                  value={userSearch}
                  onChange={(e) => setUserSearch(e.target.value)}
                  id="user-search-input"
                />
              </div>
              <button className="refresh-btn" onClick={fetchUsers} id="refresh-users-btn">🔄 Refresh</button>
            </div>

            {loadingUsers ? <SkeletonRows n={6} /> : (
              <div className="table-wrapper">
                <table className="admin-table">
                  <thead>
                    <tr>
                      <th>#</th><th>Username</th><th>Email</th><th>Joined</th><th>Role</th><th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredUsers.map((u, i) => (
                      <tr key={u._id} className={confirmDelete === u._id ? "row-danger" : ""}>
                        <td className="muted">{i + 1}</td>
                        <td className="bold">{u.username}</td>
                        <td className="muted">{u.email}</td>
                        <td className="muted">
                          {u.createdAt ? new Date(u.createdAt).toLocaleDateString("en-IN") : "—"}
                        </td>
                        <td><span className={`role-badge role-${u.role}`}>{u.role}</span></td>
                        <td>
                          <div className="action-group">
                            {/* Role toggle */}
                            {u.role !== "admin" ? (
                              <button
                                className="btn-role-sm"
                                onClick={() => handleToggleRole(u)}
                                disabled={togglingRole === u._id}
                                id={`toggle-role-${u._id}`}
                                title="Make admin"
                              >
                                {togglingRole === u._id ? "…" : "⬆️ Make Admin"}
                              </button>
                            ) : (
                              <button
                                className="btn-role-sm btn-demote"
                                onClick={() => handleToggleRole(u)}
                                disabled={togglingRole === u._id}
                                id={`demote-role-${u._id}`}
                                title="Demote to user"
                              >
                                {togglingRole === u._id ? "…" : "⬇️ Demote"}
                              </button>
                            )}

                            {/* Delete */}
                            {confirmDelete === u._id ? (
                              <div className="confirm-row">
                                <span className="confirm-text">Sure?</span>
                                <button className="btn-danger-sm" onClick={() => handleDeleteUser(u._id)} id={`confirm-del-user-${u._id}`}>Yes</button>
                                <button className="btn-cancel-sm" onClick={() => setConfirmDelete(null)}>No</button>
                              </div>
                            ) : (
                              <button className="btn-danger-sm" onClick={() => setConfirmDelete(u._id)} id={`delete-user-${u._id}`}>🗑️</button>
                            )}
                          </div>
                        </td>
                      </tr>
                    ))}
                    {filteredUsers.length === 0 && (
                      <tr><td colSpan={6} className="empty-row">
                        {userSearch ? "No users match your search" : "No users registered yet"}
                      </td></tr>
                    )}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        )}

        {/* ════════ AUCTIONS ════════ */}
        {activeTab === "auctions" && (
          <section className="content-section">
            <div className="section-toolbar">
              <button className="btn-primary" onClick={() => setShowAuctionForm((v) => !v)} id="toggle-auction-form-btn">
                {showAuctionForm ? "✕ Cancel" : "＋ Add Auction"}
              </button>
              <button className="refresh-btn" onClick={fetchAuctions} id="refresh-auctions-btn">🔄 Refresh</button>
            </div>

            {showAuctionForm && (
              <div className="auction-form-card">
                <h3 className="form-title">➕ New Auction Listing</h3>
                <form onSubmit={handleAddAuction} className="auction-form" id="add-auction-form">
                  <div className="form-grid">
                    <div className="form-group">
                      <label>Crop / Item Name *</label>
                      <input type="text" placeholder="e.g. Organic Wheat" value={auctionForm.name}
                        onChange={(e) => setAuctionForm({ ...auctionForm, name: e.target.value })} required id="auction-name-input" />
                    </div>
                    <div className="form-group">
                      <label>Price (₹)</label>
                      <input type="number" placeholder="e.g. 2500" value={auctionForm.price}
                        onChange={(e) => setAuctionForm({ ...auctionForm, price: e.target.value })} id="auction-price-input" />
                    </div>
                    <div className="form-group">
                      <label>Quantity (kg / units)</label>
                      <input type="number" placeholder="e.g. 100" value={auctionForm.quantity}
                        onChange={(e) => setAuctionForm({ ...auctionForm, quantity: e.target.value })} id="auction-qty-input" />
                    </div>
                    <div className="form-group full-width">
                      <label>Description *</label>
                      <textarea placeholder="Brief description…" value={auctionForm.description}
                        onChange={(e) => setAuctionForm({ ...auctionForm, description: e.target.value })}
                        required rows={3} id="auction-desc-input" />
                    </div>
                  </div>
                  <button type="submit" className="btn-primary" disabled={addingAuction} id="submit-auction-btn">
                    {addingAuction ? "Adding…" : "✅ Add Listing"}
                  </button>
                </form>
              </div>
            )}

            {loadingAuctions ? <SkeletonRows n={5} /> : (
              <div className="auction-cards-grid">
                {auctions.map((a) => (
                  <div className="auction-card" key={a._id}>
                    <div className="auction-card-header">
                      <h3 className="auction-name">{a.name}</h3>
                      {confirmDelete === a._id ? (
                        <div className="confirm-inline">
                          <button className="btn-danger-sm" onClick={() => handleDeleteAuction(a._id)} id={`confirm-del-auction-${a._id}`}>✓</button>
                          <button className="btn-cancel-sm" onClick={() => setConfirmDelete(null)}>✕</button>
                        </div>
                      ) : (
                        <button className="btn-danger-sm" onClick={() => setConfirmDelete(a._id)} id={`delete-auction-${a._id}`} title="Delete">🗑️</button>
                      )}
                    </div>
                    <p className="auction-desc">{a.description}</p>
                    <div className="auction-meta">
                      {a.price    && <span className="meta-chip price-chip">💰 ₹{Number(a.price).toLocaleString()}</span>}
                      {a.quantity && <span className="meta-chip qty-chip">📦 {a.quantity} units</span>}
                    </div>
                    {a.createdAt && (
                      <p className="auction-date">🗓️ {new Date(a.createdAt).toLocaleDateString("en-IN")}</p>
                    )}
                  </div>
                ))}
                {auctions.length === 0 && (
                  <div className="empty-state">
                    <div className="empty-icon">🏷️</div>
                    <p>No auctions yet. Add one above!</p>
                  </div>
                )}
              </div>
            )}
          </section>
        )}
      </main>
    </div>
  );
}

// ── Skeleton helper ──────────────────────────────────────────
function SkeletonRows({ n = 4 }) {
  return (
    <div className="skeleton-list">
      {Array.from({ length: n }).map((_, i) => (
        <div key={i} className="skeleton-row" />
      ))}
    </div>
  );
}

export default AdminDashboard;
